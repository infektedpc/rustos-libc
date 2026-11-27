//! rustos-libc — the userspace runtime (libc) for RustOS.
//!
//! * `syscall` — raw syscall layer: the nine numbers (verified
//!   against `rustos-process/src/syscall.rs`), the errno [`Error`]
//!   type, the [`SyscallBackend`] trait, [`RealBackend`] (the
//!   x86_64 `syscall` instruction — the default backend since
//!   Batch 11 wired the kernel's MSR entry), [`Int80Backend`]
//!   (the `int 0x80` trap — the fallback, selected by the
//!   `int80-backend` feature) and [`MockBackend`] (records calls,
//!   answers from a scripted queue, captures stdout/stderr writes,
//!   open paths and spawn arguments for tests). [`default_backend()`]
//!   is the constructor real programs call.
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
//! A program using this crate is `#![no_std]` + `#![no_main]` and
//! provides its own `_start` and panic handler. It does NOT declare
//! a global allocator of its own: this crate provides it —
//! [`ALLOCATOR`] is the global allocator on a freestanding target
//! when the program enables the `global-allocator` feature (see
//! its docs) — and [`start`] installs the kernel-handed heap
//! region into it. Declaring a second `#[global_allocator]` static
//! in the program would leave `format!`/`Vec`/`String` allocating
//! through a never-initialised heap, which fails every allocation
//! (Batch 9: both live programs exited 1 before printing).
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
    check, Error, Int80Backend, MockBackend, RealBackend, Result, SyscallBackend, default_backend,
    SyscallCall, ALL_SYSCALLS, FD_STDERR, FD_STDIN, FD_STDOUT, SYS_CLOSE, SYS_EXIT, SYS_GETPID,
    SYS_OPEN, SYS_READ, SYS_SPAWN, SYS_UPTIME, SYS_WAIT, SYS_WRITE, SYS_YIELD,
};

/// Convenience re-exports so userspace programs need only this
/// crate for the usual heap types.
pub use alloc::{boxed::Box, string::String, vec::Vec};
pub use alloc::format;

/// Print to stdout through `sys`, using `core::fmt` formatting.
/// `sys` is `&mut Syscalls` (e.g. `uprint!(&mut sys, "pid={}", pid)`).
///
/// The whole formatted line goes out in ONE `write` syscall. The
/// old version formatted through `FileWriter`, which issues one
/// syscall per `core::fmt` fragment (literal, argument, literal,
/// …), so a timer tick between two fragments let another task's
/// output land mid-line — the Batch 8 live boot showed `hello from
/// userspace (pid 2)` and the kernel's `userspace: hello task pid 2
/// created` on the same line for exactly this reason. Formatting
/// first (heap `String`, installed by `start()` before `main`) and
/// writing once makes a line atomic: the kernel's `write_console`
/// holds the console lock for the whole buffer.
#[macro_export]
macro_rules! uprint {
    ($sys:expr, $($arg:tt)*) => {
        {
            let _ = $sys.write_str($crate::FD_STDOUT, &$crate::format!($($arg)*));
        }
    };
}

/// Like [`uprint!`], plus a trailing newline — still a single
/// `write` syscall (see [`uprint!`]).
#[macro_export]
macro_rules! uprintln {
    ($sys:expr, $($arg:tt)*) => {
        {
            let mut s = $crate::format!($($arg)*);
            s.push('\n');
            let _ = $sys.write_str($crate::FD_STDOUT, &s);
        }
    };
    ($sys:expr) => {
        {
            let _ = $sys.write_str($crate::FD_STDOUT, "\n");
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
        // One `write` syscall per macro call, not one per `core::fmt`
        // fragment: the Batch 8 live boot interleaved another task's
        // line inside a multi-write `uprintln!`. A line that goes
        // out in one write cannot be split by a timer tick.
        assert_eq!(mock.calls_for(SYS_WRITE).len(), 3);
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
