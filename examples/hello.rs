//! Example RustOS userspace program: prints a greeting and exits.
//!
//! It uses only `rustos_libc` (+ `core`/`alloc`): one `write` to
//! stdout, then `exit(0)`. It spawns nothing.
//!
//! On a Linux x86_64 host this happens to work for real, because
//! Linux shares syscall numbers 1 (`write`) and 60 (`exit`) with
//! RustOS — running it prints `Hello, RustOS!` to the host terminal.
//! On RustOS the same binary talks to the RustOS kernel instead.
//!
//! A real RustOS program would be `#![no_std]` + `#![no_main]` with
//! its own `_start` calling `rustos_libc::start(heap, main)`; this
//! example keeps a host `main` so it can be built and run here.
//! The program logic itself is in [`run`], which takes a
//! [`Syscalls`] over any backend — the test below drives it through
//! a [`MockBackend`] so no real syscall is executed.

use rustos_libc::{Error, FD_STDOUT, RealBackend, Result, Syscalls};

/// The program logic: write the greeting to stdout.
pub fn run(sys: &mut Syscalls) -> Result<usize> {
    sys.write_str(FD_STDOUT, "Hello, RustOS!\n")
}

fn main() {
    let mut backend = RealBackend;
    let mut sys = Syscalls::new(&mut backend);
    let code = match run(&mut sys) {
        Ok(_) => 0,
        Err(Error(_)) => 1,
    };
    sys.exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustos_libc::{MockBackend, SYS_WRITE};

    #[test]
    fn hello_writes_greeting_and_nothing_else() {
        let mut mock = MockBackend::new();
        // The write reports its full length ("Hello, RustOS!\n" is 15 bytes).
        mock.push_returns(&[15]);
        let mut sys = Syscalls::new(&mut mock);
        assert_eq!(run(&mut sys), Ok(15));
        drop(sys);
        assert_eq!(mock.calls.len(), 1);
        assert_eq!(mock.calls[0].number, SYS_WRITE);
        assert_eq!(mock.calls[0].a0, FD_STDOUT);
        assert_eq!(mock.output, b"Hello, RustOS!\n");
    }

    #[test]
    fn hello_reports_write_errors() {
        let mut mock = MockBackend::new();
        mock.push_returns(&[-9]); // EBADF
        let mut sys = Syscalls::new(&mut mock);
        assert_eq!(run(&mut sys), Err(Error::BADF));
    }
}
