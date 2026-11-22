//! rustos-libc — the userspace runtime (libc) for RustOS.
//!
//! * `syscall` — raw syscall layer: the nine numbers (verified
//!   against `rustos-process/src/syscall.rs`), the errno [`Error`]
//!   type, the [`SyscallBackend`] trait, [`RealBackend`] (the
//!   x86_64 `syscall` instruction) and [`MockBackend`] (records
//!   calls, answers from a scripted queue, captures stdout/stderr
//!   writes, open paths and spawn arguments for tests).
//! * `io` — [`Syscalls`]: typed wrappers (`read`/`write`/`open`/
//!   `close`/`yield_now`/`getpid`/`spawn`/`wait`/`exit`) that
//!   translate negative returns into [`Error`] — callers never see
//!   a raw negative number.
//! * `heap` — [`Heap`]: a first-fit free-list allocator over a
//!   caller-provided region (split + coalesce, double-free
//!   detection), and [`RustosAlloc`]: a `GlobalAlloc` over one heap
//!   behind a spinlock, so programs can use `Vec`/`String`/`Box`.
//! * `fmt` — `printf`-style formatting ([`format_printf`],
//!   [`printf_to`], [`pargs!`]) with `%d %i %u %x %X %s %c %%`
//!   plus width / left-align / zero-pad, rendered through
//!   `core::fmt`.
//! * `file` — [`File`]: seek-less file I/O over fds (there is no
//!   `lseek` syscall yet), stdin/stdout/stderr handles, and
//!   [`FileWriter`]: a `core::fmt::Write` sink for one fd.
//! * `start` — program startup: [`start`]/[`init_runtime`] install
//!   the kernel-handed heap region and run `main`, then exit.
//!
//! The crate is `#![no_std]` outside tests (`core` + `alloc` only,
//! no external crates) so it can link into a userspace RustOS
//! program. Tests run on the host with std, against
//! [`MockBackend`] — no real syscall is ever executed on the host
//! by the test suite.
//!
//! A program using this crate is `#![no_std]` + `#![no_main]`,
//! provides its own `_start` and panic handler, and declares
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOC: RustosAlloc = RustosAlloc::new();
//! ```
//!
//! See `examples/hello.rs` and the README for the full shape.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod heap;
pub mod file;
pub mod fmt;
pub mod io;
pub mod start;
pub mod syscall;

pub use crate::heap::{init_heap, Heap, RustosAlloc, ALLOCATOR};
pub use file::{File, FileWriter};
pub use fmt::{format_printf, printf_to, IntoPrintfArg, PrintfArg};
pub use io::{Syscalls, O_APPEND, O_CREAT, O_RDONLY, O_RDWR, O_TRUNC, O_WRONLY};
pub use start::{heap_free_bytes, heap_ready, init_runtime, start, StartInfo};
pub use syscall::{
    check, Error, MockBackend, RealBackend, Result, SyscallBackend, SyscallCall,
    ALL_SYSCALLS, FD_STDERR, FD_STDIN, FD_STDOUT, SYS_CLOSE, SYS_EXIT, SYS_GETPID, SYS_OPEN,
    SYS_READ, SYS_SPAWN, SYS_WAIT, SYS_WRITE, SYS_YIELD,
};

/// Convenience re-exports so userspace programs need only this
/// crate for the usual heap types.
pub use alloc::{boxed::Box, string::String, vec::Vec};
pub use alloc::format;

/// Print to stdout through `sys`, using `core::fmt` formatting.
/// `sys` is `&mut Syscalls` (e.g. `uprint!(&mut sys, "pid={}", pid)`).
#[macro_export]
macro_rules! uprint {
    ($sys:expr, $($arg:tt)*) => {
        {
            use core::fmt::Write as _;
            let _ = write!(
                $crate::FileWriter::new($crate::FD_STDOUT, $sys.backend()),
                $($arg)*
            );
        }
    };
}

/// Like [`uprint!`], plus a trailing newline.
#[macro_export]
macro_rules! uprintln {
    ($sys:expr, $($arg:tt)*) => {
        {
            use core::fmt::Write as _;
            let _ = write!(
                $crate::FileWriter::new($crate::FD_STDOUT, $sys.backend()),
                $($arg)*
            );
            let _ = write!(
                $crate::FileWriter::new($crate::FD_STDOUT, $sys.backend()),
                "\n"
            );
        }
    };
    ($sys:expr) => {
        {
            use core::fmt::Write as _;
            let _ = write!(
                $crate::FileWriter::new($crate::FD_STDOUT, $sys.backend()),
                "\n"
            );
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syscall::MockBackend;

    #[test]
    fn uprint_macros_write_to_stdout_via_mock() {
        let mut mock = MockBackend::new();
        mock.push_returns(&[64, 64, 64, 64]);
        let mut sys = Syscalls::new(&mut mock);
        uprint!(&mut sys, "pid={}", 42);
        uprintln!(&mut sys, "done");
        uprintln!(&mut sys);
        drop(sys);
        assert_eq!(mock.output, b"pid=42done\n\n");
    }

    #[test]
    fn crate_root_reexports_line_up() {
        assert_eq!(SYS_READ, 0);
        assert_eq!(SYS_EXIT, 60);
        assert_eq!(FD_STDOUT, 1);
        let v: Vec<u8> = alloc::vec![1, 2, 3];
        assert_eq!(v.len(), 3);
        let s: String = format!("x={}", 1);
        assert_eq!(s, "x=1");
    }
}
