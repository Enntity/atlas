// SPDX-License-Identifier: AGPL-3.0-only

//! Raw kernel launch: [`AtlasRegistry::launch_on_stream`] and the programmatic
//! dependent launch (PDL) plumbing it uses for [`mark_pdl`]-ed functions, plus
//! cooperative launch ([`AtlasRegistry::launch_cooperative_on_stream`]) and its
//! occupancy query.

use std::ffi::c_void;

use cudarc::driver::LaunchConfig;

use super::{AtlasRegistry, RawCudaFunc, cuFuncSetAttribute, cuLaunchKernel, cuda_error_text};
use crate::error::{AtlasError, Result};

/// `CUlaunchAttribute` (driver API): id, padding, 64-byte value union.
#[repr(C)]
struct CuLaunchAttribute {
    id: u32,
    pad: [u8; 4],
    value: [u8; 64],
}

/// `CUlaunchConfig` (driver API).
#[repr(C)]
struct CuLaunchConfig {
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem_bytes: u32,
    stream: *mut c_void,
    attrs: *mut CuLaunchAttribute,
    num_attrs: u32,
}

unsafe extern "C" {
    fn cuLaunchKernelEx(
        config: *const CuLaunchConfig,
        f: *mut c_void,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks: *mut i32,
        func: *mut c_void,
        blockSize: i32,
        dynamicSMemSize: usize,
    ) -> i32;
    fn cuLaunchCooperativeKernel(
        f: *mut c_void,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        hStream: *mut c_void,
        kernelParams: *mut *mut c_void,
    ) -> i32;
}

const CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION: u32 = 6;

/// Functions launched with programmatic stream serialization (PDL): each one
/// runs `atlas_pdl_enter()` (kernels/gb10/common/atlas_pdl.cuh) first, so it
/// may be scheduled before its stream predecessor finishes.
static PDL_FUNCS: std::sync::RwLock<Vec<usize>> = std::sync::RwLock::new(Vec::new());

/// Launch `func` with programmatic dependent launch from now on.
pub fn mark_pdl(func: RawCudaFunc) {
    let mut funcs = PDL_FUNCS.write().unwrap_or_else(|e| e.into_inner());
    if !funcs.contains(&(func.0 as usize)) {
        funcs.push(func.0 as usize);
    }
}

fn is_pdl(func: RawCudaFunc) -> bool {
    PDL_FUNCS
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .contains(&(func.0 as usize))
}

impl AtlasRegistry {
    /// Launch a kernel on a specified raw CUDA stream.
    ///
    /// When `stream_ptr` comes from the caller (e.g. `torch.cuda.current_stream().cuda_stream`),
    /// this ensures kernels are captured during CUDA graph recording.
    ///
    /// # Safety
    /// - `kernel_params` must contain valid pointers to arguments matching the kernel signature.
    /// - `stream_ptr` must be a valid CUstream handle (or 0 to use Atlas's own stream).
    /// - `raw_func` must be a valid CUfunction obtained from `raw_function_cached`.
    pub unsafe fn launch_on_stream(
        &self,
        raw_func: RawCudaFunc,
        cfg: LaunchConfig,
        stream_ptr: u64,
        kernel_params: &mut [*mut c_void],
    ) -> Result<()> {
        // Always use the caller's stream directly. When stream_ptr=0, CUDA
        // treats it as the legacy default stream which has implicit
        // synchronization with all other streams in the same context.
        // Never fall back to Atlas's private stream — that breaks ordering
        // with PyTorch operations and prevents CUDA graph capture.
        let stream = stream_ptr;
        // Opt in to >48KB dynamic shared memory when requested.
        if cfg.shared_mem_bytes > 48 * 1024 {
            const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
            let attr_status = unsafe {
                cuFuncSetAttribute(
                    raw_func.0,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    cfg.shared_mem_bytes as i32,
                )
            };
            if attr_status != 0 {
                return Err(AtlasError::KernelLaunch(format!(
                    "cuFuncSetAttribute(MAX_DYNAMIC_SHARED={}) failed: {}",
                    cfg.shared_mem_bytes,
                    cuda_error_text(attr_status)
                )));
            }
        }
        let status = if is_pdl(raw_func) {
            let mut value = [0u8; 64];
            value[..4].copy_from_slice(&1i32.to_ne_bytes());
            let mut attr = CuLaunchAttribute {
                id: CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
                pad: [0; 4],
                value,
            };
            let config = CuLaunchConfig {
                grid: [cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2],
                block: [cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2],
                shared_mem_bytes: cfg.shared_mem_bytes,
                stream: stream as *mut c_void,
                attrs: &mut attr,
                num_attrs: 1,
            };
            unsafe {
                cuLaunchKernelEx(
                    &config,
                    raw_func.0,
                    kernel_params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            }
        } else {
            unsafe {
                cuLaunchKernel(
                    raw_func.0,
                    cfg.grid_dim.0,
                    cfg.grid_dim.1,
                    cfg.grid_dim.2,
                    cfg.block_dim.0,
                    cfg.block_dim.1,
                    cfg.block_dim.2,
                    cfg.shared_mem_bytes,
                    stream as *mut c_void,
                    kernel_params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            }
        };
        if status != 0 {
            return Err(AtlasError::KernelLaunch(format!(
                "cuLaunchKernel failed: {} (grid=[{},{},{}], block=[{},{},{}], shared_mem={})",
                cuda_error_text(status),
                cfg.grid_dim.0,
                cfg.grid_dim.1,
                cfg.grid_dim.2,
                cfg.block_dim.0,
                cfg.block_dim.1,
                cfg.block_dim.2,
                cfg.shared_mem_bytes
            )));
        }
        Ok(())
    }

    /// Cooperative launch (`cuLaunchCooperativeKernel`): every CTA of the grid
    /// is guaranteed co-resident, which kernels that spin on inter-CTA locks
    /// require. Opts in to >48KB dynamic shared memory like
    /// [`Self::launch_on_stream`]. A grid larger than the device can hold
    /// resident fails synchronously (CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE)
    /// without enqueuing work.
    ///
    /// # Safety
    /// Same contract as [`Self::launch_on_stream`].
    pub unsafe fn launch_cooperative_on_stream(
        &self,
        raw_func: RawCudaFunc,
        cfg: LaunchConfig,
        stream_ptr: u64,
        kernel_params: &mut [*mut c_void],
    ) -> Result<()> {
        if cfg.shared_mem_bytes > 48 * 1024 {
            const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
            let attr_status = unsafe {
                cuFuncSetAttribute(
                    raw_func.0,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    cfg.shared_mem_bytes as i32,
                )
            };
            if attr_status != 0 {
                return Err(AtlasError::KernelLaunch(format!(
                    "cuFuncSetAttribute(MAX_DYNAMIC_SHARED={}) failed: {}",
                    cfg.shared_mem_bytes,
                    cuda_error_text(attr_status)
                )));
            }
        }
        let status = unsafe {
            cuLaunchCooperativeKernel(
                raw_func.0,
                cfg.grid_dim.0,
                cfg.grid_dim.1,
                cfg.grid_dim.2,
                cfg.block_dim.0,
                cfg.block_dim.1,
                cfg.block_dim.2,
                cfg.shared_mem_bytes,
                stream_ptr as *mut c_void,
                kernel_params.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(AtlasError::KernelLaunch(format!(
                "cuLaunchCooperativeKernel failed: {} (grid=[{},{},{}], block=[{},{},{}], shared_mem={})",
                cuda_error_text(status),
                cfg.grid_dim.0,
                cfg.grid_dim.1,
                cfg.grid_dim.2,
                cfg.block_dim.0,
                cfg.block_dim.1,
                cfg.block_dim.2,
                cfg.shared_mem_bytes
            )));
        }
        Ok(())
    }

    /// Resident CTAs per SM for a cooperative launch of `raw_func` with
    /// `block` threads and `smem` dynamic bytes. First sets the function's
    /// preferred shared-memory carveout to the maximum (100%) and, above
    /// 48KB, its max dynamic shared size, so the occupancy reported is the
    /// one the launch gets.
    pub fn cooperative_blocks_per_sm(
        &self,
        raw_func: RawCudaFunc,
        block: u32,
        smem: u32,
    ) -> Result<u32> {
        const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
        const CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: i32 = 9;
        let mut attrs = vec![(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100)];
        if smem > 48 * 1024 {
            attrs.push((CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32));
        }
        for (attr, value) in attrs {
            let status = unsafe { cuFuncSetAttribute(raw_func.0, attr, value) };
            if status != 0 {
                return Err(AtlasError::KernelLaunch(format!(
                    "cuFuncSetAttribute({attr}={value}) failed: {}",
                    cuda_error_text(status)
                )));
            }
        }
        let mut blocks: i32 = 0;
        let status = unsafe {
            cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &mut blocks,
                raw_func.0,
                block as i32,
                smem as usize,
            )
        };
        if status != 0 {
            return Err(AtlasError::KernelLaunch(format!(
                "cuOccupancyMaxActiveBlocksPerMultiprocessor failed: {}",
                cuda_error_text(status)
            )));
        }
        Ok(blocks.max(0) as u32)
    }
}
