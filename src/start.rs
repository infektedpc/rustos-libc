//! Program startup: what the kernel hands a program, and what this
//! crate does with it.
//!
//! ## The kernel contract (what a program's `_start` gets)
//!
//! The RustOS kernel loads a userspace program and hands it exactly
//! two things before jumping to its entry point:
//!
//! * a heap region: a base address and a size, and
//! * standard file descriptors 0/1/2 already open (stdin/stdout/stderr).
//!
//! How the region is communicated (registers, a struct at a fixed
//! address, the initial stack) is **not yet defined by the kernel** —
//! `rustos-process` loads ELF segments and builds address spaces but
//! has no heap-handoff ABI yet. Until the kernel defines it, a
//! program's `_start` must obtain the region by whatever the kernel
//! of the day provides and pass it here. This module is the
//! receiving end of that contract, not the kernel side.
//!
//! ## What this crate provides
//!
//! * [`init_runtime`] — install the heap region into the global
//!   [`crate::ALLOCATOR`] (backing `Vec`/`String`/`Box` once the
//!   program sets `#[global_allocator]`).
//! * [`start`] — initialise the heap, run `main`, and `exit` with
//!   its return code. This is the whole runtime: there is no
//!   argument parsing, no environment, no TLS, no atexit.
//! * [`StartInfo`] — the heap region, as one value.
//!
//! ## What a program must provide
//!
//! A userspace program is `#![no_std]` + `#![no_main]` and defines
//! its own `_start` (the ELF entry point) and a panic handler —
//! this crate provides neither, because the panic strategy
//! (abort? print-and-exit?) belongs to the program. Its `_start`
//! obtains the heap region from the kernel, then calls
//! [`start`] with its `main` function. See `examples/hello.rs`
//! for the shape (the example runs on the host with
//! [`crate::RealBackend`]; a real program's `_start` would call
//! `start` instead of returning to a host `main`).

use crate::heap::{init_heap, ALLOCATOR};
use crate::io::Syscalls;
use crate::syscall::RealBackend;

/// The region the kernel hands a userspace program at startup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StartInfo {
    /// Base address of the heap region.
    pub heap_base: *mut u8,
    /// Size of the heap region in bytes.
    pub heap_size: usize,
}

impl StartInfo {
    pub const fn new(heap_base: *mut u8, heap_size: usize) -> Self {
        Self {
            heap_base,
            heap_size,
        }
    }
}

/// Initialise the runtime from a [`StartInfo`]: install the heap
/// region into the global allocator. Returns `true` if this call
/// installed the heap, `false` if it was already installed.
///
/// # Safety
///
/// The region must satisfy [`Heap::new`]'s requirements (writable,
/// owned by this process, alive for the process lifetime).
///
/// [`Heap::new`]: crate::Heap::new
pub unsafe fn init_runtime(info: StartInfo) -> bool {
    init_heap(info.heap_base, info.heap_size)
}

/// Run a userspace program: install the heap, call `main`, and exit
/// with its return code. Never returns.
///
/// `main` runs with the global heap installed, so it may use
/// `alloc::vec::Vec`/`String`/`Box` provided the program declared
/// `#[global_allocator] static ALLOC: RustosAlloc =
/// RustosAlloc::new();`.
///
/// # Safety
///
/// The heap region must satisfy [`Heap::new`]'s requirements.
///
/// [`Heap::new`]: crate::Heap::new
pub fn start(info: StartInfo, main: fn() -> i32) -> ! {
    unsafe {
        init_runtime(info);
    }
    let code = main();
    let mut backend = RealBackend;
    let mut sys = Syscalls::new(&mut backend);
    sys.exit(code)
}

/// True once the global heap has been installed.
pub fn heap_ready() -> bool {
    ALLOCATOR.is_initialized()
}

/// Bytes currently free in the global heap (0 if not installed).
pub fn heap_free_bytes() -> usize {
    ALLOCATOR.free_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_info_round_trips() {
        let info = StartInfo::new(0x1000 as *mut u8, 4096);
        assert_eq!(info.heap_base as usize, 0x1000);
        assert_eq!(info.heap_size, 4096);
    }

    #[test]
    fn heap_ready_reflects_allocator_state() {
        // ALLOCATOR may already be initialised by the heap test in
        // this same process; either state is fine, but the accessors
        // must agree with the allocator itself.
        assert_eq!(heap_ready(), ALLOCATOR.is_initialized());
        assert_eq!(heap_free_bytes(), ALLOCATOR.free_bytes());
    }
}
