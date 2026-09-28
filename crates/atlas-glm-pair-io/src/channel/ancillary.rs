// SPDX-License-Identifier: AGPL-3.0-only

use super::Credentials;
use std::io;
use std::mem::size_of;

pub(super) const CAPACITY: usize = unsafe {
    libc::CMSG_SPACE(size_of::<libc::ucred>() as _) as usize
        + libc::CMSG_SPACE((16 * size_of::<i32>()) as _) as usize
};

#[repr(C)]
pub(super) union Storage {
    align: libc::cmsghdr,
    bytes: [u8; CAPACITY],
}

impl Storage {
    pub(super) fn new() -> Self {
        Self {
            bytes: [0; CAPACITY],
        }
    }
    pub(super) fn as_mut_ptr(&mut self) -> *mut libc::c_void {
        std::ptr::from_mut(self).cast()
    }
}

/// Only a kernel-populated recvmsg header backed by `Storage` may be passed.
/// Do not return early on an invalid record: later records may own FDs.
pub(super) unsafe fn validate_and_dispose(
    message: &libc::msghdr,
    expected: Credentials,
) -> io::Result<()> {
    let mut bad = message.msg_flags & (libc::MSG_TRUNC | libc::MSG_CTRUNC) != 0;
    let mut credentials = None;
    let mut header = libc::CMSG_FIRSTHDR(message);
    while !header.is_null() {
        let length = (*header).cmsg_len;
        let offset = (header as usize).saturating_sub(message.msg_control as usize);
        let base = libc::CMSG_LEN(0) as usize;
        // The kernel supplies structurally valid records even on truncation.
        // Keep pointer reads bounded regardless; malformed structure is fatal.
        if length < base
            || offset
                .checked_add(length)
                .is_none_or(|end| end > message.msg_controllen)
        {
            bad = true;
            break;
        }
        let data_len = length - base;
        if (*header).cmsg_level == libc::SOL_SOCKET && (*header).cmsg_type == libc::SCM_RIGHTS {
            bad = true;
            let data = libc::CMSG_DATA(header).cast::<i32>();
            for index in 0..data_len / size_of::<i32>() {
                // Payload alignment is not guaranteed by CMSG_DATA.
                let fd = std::ptr::read_unaligned(data.add(index));
                if fd >= 0 {
                    libc::close(fd);
                }
            }
        } else if (*header).cmsg_level == libc::SOL_SOCKET
            && (*header).cmsg_type == libc::SCM_CREDENTIALS
            && data_len == size_of::<libc::ucred>()
        {
            let value = std::ptr::read_unaligned(libc::CMSG_DATA(header).cast::<libc::ucred>());
            if credentials.is_some() {
                bad = true;
            }
            credentials = Some(Credentials {
                pid: value.pid,
                uid: value.uid,
                gid: value.gid,
            });
        } else {
            bad = true;
        }
        header = libc::CMSG_NXTHDR(message, header);
    }
    if bad || credentials != Some(expected) {
        Err(io::Error::other(
            "invalid paired credentials or ancillary data",
        ))
    } else {
        Ok(())
    }
}
