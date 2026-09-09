// SPDX-License-Identifier: AGPL-3.0-only
//! Bounded exec inputs. Canonical validation is not recipe authorization.
use super::*;
use std::ffi::CString;

pub struct Spec {
    pub(super) executable: OwnedFd,
    pub(super) argv: Vec<CString>,
    pub(super) env: Vec<CString>,
    pub(super) explicit_env: bool,
}
impl Spec {
    pub fn new(args: &[String]) -> io::Result<Self> {
        if args.is_empty() || args.len() > 64 || args.iter().any(|x| x.len() > 4096) {
            return Err(error("bounded executable and argv required"));
        }
        let argv: Vec<_> = args
            .iter()
            .map(|s| CString::new(s.as_bytes()))
            .collect::<Result<_, _>>()
            .map_err(|_| error("NUL in argument"))?;
        let executable = owned(unsafe {
            libc::open(
                argv[0].as_ptr(),
                libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        })?;
        let mut stat = unsafe { std::mem::zeroed::<libc::stat>() };
        if unsafe { libc::fstat(executable.as_raw_fd(), &mut stat) } != 0
            || stat.st_mode & libc::S_IFMT != libc::S_IFREG
            || stat.st_mode & 0o6000 != 0
            || stat.st_mode & 0o111 == 0
        {
            return Err(error(
                "executable must be regular, executable and not set-ID",
            ));
        }
        let mut magic = [0u8; 4];
        if unsafe { libc::pread(executable.as_raw_fd(), magic.as_mut_ptr().cast(), 4, 0) } != 4
            || magic != *b"\x7fELF"
        {
            return Err(error("ELF executable required; no scripts"));
        }
        let cap = unsafe {
            libc::fgetxattr(
                executable.as_raw_fd(),
                c"security.capability".as_ptr(),
                std::ptr::null_mut(),
                0,
            )
        };
        if cap >= 0 || io::Error::last_os_error().raw_os_error() != Some(libc::ENODATA) {
            return Err(error(
                "executable capability check failed or capability present",
            ));
        }
        Ok(Self {
            executable,
            argv,
            env: vec![CString::new("PATH=/usr/bin:/bin").unwrap()],
            explicit_env: false,
        })
    }
    /// No ambient environment is copied. The LIVE caller must independently
    /// match this entire table to the authorized canonical recipe, including
    /// reviewed library paths and required absence of other settings.
    pub fn with_environment(args: &[String], environment: &[(String, String)]) -> io::Result<Self> {
        if environment.is_empty() || environment.len() > 128 {
            return Err(error("bounded explicit LIVE environment required"));
        }
        let mut previous: Option<&str> = None;
        let mut bytes = 0usize;
        let mut env = Vec::with_capacity(environment.len());
        for (key, value) in environment {
            if key.is_empty()
                || key.len() > 4096
                || value.len() > 4096
                || !key
                    .bytes()
                    .all(|b| b.is_ascii_uppercase() || b == b'_' || b.is_ascii_digit())
                || key.as_bytes()[0].is_ascii_digit()
                || previous.is_some_and(|p| p >= key.as_str())
            {
                return Err(error(
                    "LIVE environment keys must be canonical sorted unique names",
                ));
            }
            if key.starts_with("LD_") && key != "LD_LIBRARY_PATH" {
                return Err(error("loader injection environment is forbidden"));
            }
            if key == "LD_LIBRARY_PATH"
                && value
                    .split(':')
                    .any(|p| !p.starts_with('/') || p.split('/').any(|c| c == "." || c == ".."))
            {
                return Err(error("library path requires explicit absolute directories"));
            }
            bytes = bytes
                .checked_add(key.len() + value.len() + 2)
                .ok_or_else(|| error("environment size overflow"))?;
            if bytes > 65536 {
                return Err(error("LIVE environment exceeds recipe bound"));
            }
            env.push(
                CString::new(format!("{key}={value}")).map_err(|_| error("NUL in environment"))?,
            );
            previous = Some(key);
        }
        if !environment
            .iter()
            .any(|(k, v)| k == "PATH" && v == "/usr/bin:/bin")
            || !environment
                .iter()
                .any(|(k, v)| k == "ATLAS_GLM_PAIR_FD" && v == "3")
        {
            return Err(error(
                "LIVE environment requires exact PATH and FD3 locator",
            ));
        }
        let mut spec = Self::new(args)?;
        spec.env = env;
        spec.explicit_env = true;
        Ok(spec)
    }
}
