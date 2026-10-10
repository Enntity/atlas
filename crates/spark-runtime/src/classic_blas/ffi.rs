// SPDX-License-Identifier: AGPL-3.0-only
//! Classic CUDA BLAS I/O. Dynamic binding avoids adding a dependency to legacy paths.
use super::ClassicBlasIo;
use crate::gpu::{Bf16Gemm, Bf16GemmNumerics};
use anyhow::{Result, bail, ensure};
use std::ffi::{CStr, c_void};
type Handle = *mut c_void;
type Create = unsafe extern "C" fn(*mut Handle) -> i32;
type Destroy = unsafe extern "C" fn(Handle) -> i32;
type SetStream = unsafe extern "C" fn(Handle, Handle) -> i32;
type SetMath = unsafe extern "C" fn(Handle, i32) -> i32;
type SetWorkspace = unsafe extern "C" fn(Handle, Handle, usize) -> i32;
type GetVersion = unsafe extern "C" fn(Handle, *mut i32) -> i32;
type Gemm = unsafe extern "C" fn(
    Handle,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f32,
    Handle,
    i32,
    i32,
    Handle,
    i32,
    i32,
    *const f32,
    Handle,
    i32,
    i32,
    i32,
    i32,
) -> i32;
unsafe extern "C" {
    fn cuCtxGetCurrent(context: *mut u64) -> i32;
    fn cuCtxPushCurrent_v2(context: u64) -> i32;
    fn cuCtxPopCurrent_v2(context: *mut u64) -> i32;
    fn cuMemAlloc_v2(pointer: *mut u64, bytes: usize) -> i32;
    fn cuMemFree_v2(pointer: u64) -> i32;
    fn cuStreamSynchronize(stream: u64) -> i32;
}
struct Library(Handle);
// The mapped library is immutable and remains live while any handle owner retains Arc<Io>.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}
impl Drop for Library {
    fn drop(&mut self) {
        unsafe {
            libc::dlclose(self.0);
        }
    }
}
impl Library {
    fn symbol<T: Copy>(&self, name: &CStr) -> Result<T> {
        let pointer = unsafe { libc::dlsym(self.0, name.as_ptr()) };
        ensure!(
            !pointer.is_null(),
            "classic BLAS missing export {}",
            name.to_string_lossy()
        );
        ensure!(
            std::mem::size_of::<T>() == std::mem::size_of::<Handle>(),
            "invalid function pointer ABI"
        );
        // Called only with the corresponding explicitly declared C function type.
        Ok(unsafe { std::mem::transmute_copy(&pointer) })
    }
}
pub(crate) struct CublasIo {
    _library: Library,
    _lt_library: Library,
    create: Create,
    destroy: Destroy,
    set_stream: SetStream,
    set_math: SetMath,
    set_workspace: SetWorkspace,
    gemm: Gemm,
}
fn symbol_provider(pointer: *const c_void) -> Result<String> {
    let mut binding: libc::Dl_info = unsafe { std::mem::zeroed() };
    let located = unsafe { libc::dladdr(pointer, &mut binding) };
    ensure!(
        located != 0 && !binding.dli_fname.is_null(),
        "cannot attest classic BLAS provider"
    );
    Ok(unsafe { CStr::from_ptr(binding.dli_fname) }
        .to_string_lossy()
        .into_owned())
}
fn check(status: i32, operation: &str) -> Result<()> {
    ensure!(status == 0, "{operation} failed: status {status}");
    Ok(())
}
impl CublasIo {
    pub(crate) fn load(numerics: Bf16GemmNumerics) -> Result<Self> {
        let pointer = unsafe {
            libc::dlopen(
                c"libcublas.so.13".as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        };
        if pointer.is_null() {
            let error = unsafe { libc::dlerror() };
            let message = if error.is_null() {
                "unknown dynamic loader error".into()
            } else {
                unsafe { CStr::from_ptr(error) }.to_string_lossy()
            };
            bail!("native shared classic BLAS library unavailable: {message}");
        }
        let library = Library(pointer);
        let lt_pointer = unsafe {
            libc::dlopen(
                c"libcublasLt.so.13".as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        };
        ensure!(
            !lt_pointer.is_null(),
            "coherent classic profile requires Lt provider"
        );
        let lt_library = Library(lt_pointer);
        let lt_version: unsafe extern "C" fn() -> usize =
            lt_library.symbol(c"cublasLtGetVersion")?;
        let lt_matmul: *mut c_void = lt_library.symbol(c"cublasLtMatmul")?;
        let lt_provider = symbol_provider(lt_matmul)?;
        ensure!(
            lt_provider == symbol_provider(lt_version as *const () as *const c_void)?,
            "Lt version and GEMM providers differ"
        );
        let actual_lt_version = i32::try_from(unsafe { lt_version() })?;
        let io = Self {
            create: library.symbol(c"cublasCreate_v2")?,
            destroy: library.symbol(c"cublasDestroy_v2")?,
            set_stream: library.symbol(c"cublasSetStream_v2")?,
            set_math: library.symbol(c"cublasSetMathMode")?,
            set_workspace: library.symbol(c"cublasSetWorkspace_v2")?,
            gemm: library.symbol(c"cublasGemmEx")?,
            _library: library,
            _lt_library: lt_library,
        };
        let get_version: GetVersion = io._library.symbol(c"cublasGetVersion_v2")?;
        let provider = symbol_provider(io.gemm as *const () as *const c_void)?;
        let version_provider = symbol_provider(get_version as *const () as *const c_void)?;
        ensure!(
            provider == version_provider,
            "classic version and GEMM providers differ"
        );
        let handle = io.create()?;
        let mut version = 0;
        let status = unsafe { get_version(handle as Handle, &mut version) };
        let destroyed = unsafe { (io.destroy)(handle as Handle) };
        if destroyed != 0 {
            // A surviving library handle must retain its code mapping even on failure.
            std::mem::forget(io);
            bail!("cublasDestroy_v2 preflight failed: status {destroyed}");
        }
        check(status, "cublasGetVersion_v2")?;
        super::validate_provider_pair(
            numerics,
            version,
            actual_lt_version,
            std::path::Path::new(&provider),
            std::path::Path::new(&lt_provider),
        )?;
        tracing::info!(
            effective_version = version,
            gemm_provider = provider,
            lt_provider,
            effective_lt_version = actual_lt_version,
            "native shared classic BF16 capability loaded"
        );
        Ok(io)
    }
}
impl ClassicBlasIo for CublasIo {
    fn check_context(&self, context: u64) -> Result<()> {
        let mut current = 0;
        check(unsafe { cuCtxGetCurrent(&mut current) }, "cuCtxGetCurrent")?;
        ensure!(
            context != 0 && current == context,
            "classic BF16 GEMM CUDA context differs from backend owner"
        );
        Ok(())
    }
    fn create(&self) -> Result<u64> {
        let mut handle = std::ptr::null_mut();
        check(unsafe { (self.create)(&mut handle) }, "cublasCreate_v2")?;
        ensure!(!handle.is_null(), "classic BLAS returned null handle");
        Ok(handle as u64)
    }
    fn alloc(&self, bytes: usize) -> Result<u64> {
        let mut pointer = 0;
        check(
            unsafe { cuMemAlloc_v2(&mut pointer, bytes) },
            "classic workspace allocation",
        )?;
        ensure!(pointer != 0, "classic workspace allocation returned null");
        Ok(pointer)
    }
    fn configure(
        &self,
        handle: u64,
        stream: u64,
        workspace: u64,
        numerics: Bf16GemmNumerics,
    ) -> Result<()> {
        let (_, _, math, bytes) = numerics
            .profile()
            .ok_or_else(|| anyhow::anyhow!("classic configuration requires explicit profile"))?;
        check(
            unsafe { (self.set_stream)(handle as Handle, stream as Handle) },
            "cublasSetStream_v2",
        )?;
        // SetStream resets the workspace: bind workspace afterwards, once per owned stream.
        check(
            unsafe { (self.set_workspace)(handle as Handle, workspace as Handle, bytes) },
            "cublasSetWorkspace_v2",
        )?;
        check(
            unsafe { (self.set_math)(handle as Handle, math) },
            "cublasSetMathMode",
        )
    }
    fn launch(&self, handle: u64, request: Bf16Gemm) -> Result<()> {
        request.validate()?;
        let (compute, algorithm, _, _) = request
            .numerics
            .profile()
            .ok_or_else(|| anyhow::anyhow!("classic launch requires explicit profile"))?;
        let alpha = 1.0f32;
        let beta = 0.0f32;
        check(
            unsafe {
                (self.gemm)(
                    handle as Handle,
                    1,
                    0,
                    request.columns as i32,
                    request.rows as i32,
                    request.inner as i32,
                    &alpha,
                    request.weight.0 as Handle,
                    14,
                    request.inner as i32,
                    request.input.0 as Handle,
                    14,
                    request.inner as i32,
                    &beta,
                    request.output.0 as Handle,
                    14,
                    request.columns as i32,
                    compute,
                    algorithm,
                )
            },
            "native shared cublasGemmEx",
        )
    }
    fn synchronize(&self, context: u64, stream: u64) -> Result<()> {
        let mut current = 0;
        check(
            unsafe { cuCtxGetCurrent(&mut current) },
            "quiescence cuCtxGetCurrent",
        )?;
        let pushed = current != context;
        if pushed {
            check(
                unsafe { cuCtxPushCurrent_v2(context) },
                "classic quiescence context binding",
            )?;
        }
        let result = check(
            unsafe { cuStreamSynchronize(stream) },
            "classic quiescence stream synchronization",
        );
        let restored = if pushed {
            let mut popped = 0;
            check(
                unsafe { cuCtxPopCurrent_v2(&mut popped) },
                "classic quiescence context restore",
            )
        } else {
            Ok(())
        };
        result.and(restored)
    }
    fn cleanup(
        &self,
        context: u64,
        stream: u64,
        handle: u64,
        workspace: u64,
        pending: bool,
    ) -> Result<()> {
        let mut current = 0;
        check(
            unsafe { cuCtxGetCurrent(&mut current) },
            "cleanup cuCtxGetCurrent",
        )?;
        let pushed = current != context;
        if pushed {
            check(
                unsafe { cuCtxPushCurrent_v2(context) },
                "classic cleanup context binding",
            )?;
        }
        let result = (|| {
            if pending {
                check(
                    unsafe { cuStreamSynchronize(stream) },
                    "classic cleanup stream synchronization",
                )?;
            }
            let freed = if workspace != 0 {
                unsafe { cuMemFree_v2(workspace) }
            } else {
                0
            };
            let destroyed = if handle != 0 {
                unsafe { (self.destroy)(handle as Handle) }
            } else {
                0
            };
            check(freed, "classic workspace free")?;
            check(destroyed, "classic handle destroy")
        })();
        let restored = if pushed {
            let mut popped = 0;
            check(
                unsafe { cuCtxPopCurrent_v2(&mut popped) },
                "classic cleanup context restore",
            )
        } else {
            Ok(())
        };
        result.and(restored)
    }
}
