//! Typed wrappers over the raw syscall layer.
//!
//! [`Syscalls`] borrows a [`SyscallBackend`] and exposes one method
//! per syscall. Argument/return conventions match the kernel side
//! (`rustos-process/src/syscall.rs` `dispatch`) exactly:
//!
//! * `read(fd, buf, len)` — a0 = fd, a1 = buf pointer, a2 = len.
//! * `write(fd, buf, len)` — same shape; returns bytes written.
//! * `open(path, flags)` — a0 = pointer to a NUL-terminated path,
//!   a1 = flags; returns the new fd.
//! * `close(fd)` — a0 = fd; returns 0.
//! * `yield()` — no arguments; returns 0.
//! * `getpid()` — no arguments; returns the pid.
//! * `uptime()` — no arguments; returns milliseconds since boot
//!   (RustOS-private syscall 1000).
//! * `spawn(path, argv_ptr, argc)` — a0 = pointer to a NUL-terminated
//!   path, a1 = pointer to a NULL-terminated array of pointers to
//!   NUL-terminated argument strings, a2 = argc; returns the new pid.
//!   (The kernel dispatch currently ignores the pointers and just
//!   allocates a pid — the pointer layout here is the contract the
//!   real handler should read; see the README limits.)
//! * `exit(code)` — a0 = code; never returns.
//! * `wait(pid)` — a0 = pid; returns the child's exit code. The
//!   kernel blocks the caller until the child exits (its dispatch
//!   returns `Blocked`), so from userspace this just looks like a
//!   call that returns late.
//!
//! Negative returns are translated into [`Error`]; callers never see
//! a raw negative number.

use crate::syscall::{
    check, Error, Result, SyscallBackend, FD_STDOUT, SYS_CLOSE, SYS_EXIT, SYS_GETPID,
    SYS_OPEN, SYS_READ, SYS_SPAWN, SYS_UPTIME, SYS_WAIT, SYS_WRITE, SYS_YIELD,
};

/// `open` flags (Linux-compatible values).
pub const O_RDONLY: u32 = 0;
/// `open` flags (Linux-compatible values).
pub const O_WRONLY: u32 = 1;
/// `open` flags (Linux-compatible values).
pub const O_RDWR: u32 = 2;
/// `open` flags: create the file if it does not exist.
pub const O_CREAT: u32 = 0o100;
/// `open` flags: truncate the file to zero length.
pub const O_TRUNC: u32 = 0o1000;
/// `open` flags: all writes append at the end.
pub const O_APPEND: u32 = 0o2000;

/// Longest path (without the NUL) `open`/`spawn` will accept.
/// Longer paths are rejected with [`Error::NAMETOOLONG`].
pub const MAX_PATH_LEN: usize = 255;
/// Most arguments `spawn` will accept (not counting the NULL).
pub const MAX_ARGC: usize = 32;
/// Most bytes of argument text (including NULs) `spawn` packs.
pub const MAX_ARGV_BYTES: usize = 1024;

/// Typed syscall wrappers over a [`SyscallBackend`].
pub struct Syscalls<'a> {
    backend: &'a mut dyn SyscallBackend,
}

impl<'a> Syscalls<'a> {
    /// Wrap a backend. Real programs pass [`crate::default_backend()`]
    /// ([`crate::RealBackend`], the `syscall` instruction, since
    /// Batch 11 wired the kernel's MSR entry); tests pass
    /// [`crate::MockBackend`]. [`crate::Int80Backend`] (the
    /// `int 0x80` trap) is the fallback, selected by the
    /// `int80-backend` feature.
    pub fn new(backend: &'a mut dyn SyscallBackend) -> Self {
        Self { backend }
    }

    /// Reborrow the backend (e.g. to hand it to a [`crate::FileWriter`]).
    pub fn backend(&mut self) -> &mut dyn SyscallBackend {
        &mut *self.backend
    }

    /// One raw syscall; returns the raw `rax` value.
    pub fn raw(&mut self, number: u64, a0: u64, a1: u64, a2: u64) -> i64 {
        self.backend.syscall(number, a0, a1, a2)
    }

    /// `read(fd, buf)`: read up to `buf.len()` bytes. Returns the
    /// number of bytes read; 0 means EOF.
    pub fn read(&mut self, fd: u64, buf: &mut [u8]) -> Result<usize> {
        let ret = self.raw(SYS_READ, fd, buf.as_mut_ptr() as u64, buf.len() as u64);
        Ok(check(ret)? as usize)
    }

    /// `write(fd, buf)`: write bytes. Returns the number written,
    /// which may be less than `buf.len()` — use
    /// [`crate::File::write_all`] when the whole buffer must go out.
    pub fn write(&mut self, fd: u64, buf: &[u8]) -> Result<usize> {
        let ret = self.raw(SYS_WRITE, fd, buf.as_ptr() as u64, buf.len() as u64);
        Ok(check(ret)? as usize)
    }

    /// `write` a whole string to a file descriptor.
    pub fn write_str(&mut self, fd: u64, s: &str) -> Result<usize> {
        self.write(fd, s.as_bytes())
    }

    /// Print a string to stdout (fd 1).
    pub fn print(&mut self, s: &str) -> Result<usize> {
        self.write_str(FD_STDOUT, s)
    }

    /// `open(path, flags)`: open a file, returning its fd.
    /// The path is copied into a stack buffer and NUL-terminated
    /// before the trap; paths longer than [`MAX_PATH_LEN`] bytes are
    /// rejected with [`Error::NAMETOOLONG`] without trapping.
    pub fn open(&mut self, path: &str, flags: u32) -> Result<u64> {
        let cpath = nul_terminate(path)?;
        let ret = self.raw(SYS_OPEN, cpath.as_ptr() as u64, flags as u64, 0);
        Ok(check(ret)? as u64)
    }

    /// `close(fd)`: close a file descriptor.
    pub fn close(&mut self, fd: u64) -> Result<()> {
        check(self.raw(SYS_CLOSE, fd, 0, 0))?;
        Ok(())
    }

    /// `yield()`: voluntarily give up the CPU.
    pub fn yield_now(&mut self) -> Result<()> {
        check(self.raw(SYS_YIELD, 0, 0, 0))?;
        Ok(())
    }

    /// `getpid()`: pid of the calling process.
    pub fn getpid(&mut self) -> Result<u32> {
        Ok(check(self.raw(SYS_GETPID, 0, 0, 0))? as u32)
    }

    /// `uptime()`: milliseconds since boot (PIT ticks * 10 at
    /// 100 Hz). RustOS-private syscall 1000 (see
    /// `crate::syscall::SYS_UPTIME`); added in Batch 9 for the
    /// userspace shell's `uptime` builtin.
    pub fn uptime(&mut self) -> Result<u64> {
        Ok(check(self.raw(SYS_UPTIME, 0, 0, 0))? as u64)
    }

    /// `spawn(path, argv)`: start a new process running `path` with
    /// arguments `argv` (argv[0] is conventionally the program name).
    /// Returns the child's pid. Paths longer than [`MAX_PATH_LEN`],
    /// more than [`MAX_ARGC`] arguments, or more than
    /// [`MAX_ARGV_BYTES`] bytes of argument text are rejected with
    /// [`Error::INVAL`] without trapping.
    pub fn spawn(&mut self, path: &str, argv: &[&str]) -> Result<u32> {
        if argv.len() > MAX_ARGC {
            return Err(Error::INVAL);
        }
        let cpath = nul_terminate(path)?;
        // Pack every argument NUL-terminated into one stack buffer
        // and build the NULL-terminated pointer array beside it.
        let mut arg_bytes = [0u8; MAX_ARGV_BYTES];
        let mut arg_ptrs = [0u64; MAX_ARGC + 1];
        let mut off = 0usize;
        for (i, arg) in argv.iter().enumerate() {
            let bytes = arg.as_bytes();
            if off + bytes.len() + 1 > MAX_ARGV_BYTES {
                return Err(Error::INVAL);
            }
            arg_bytes[off..off + bytes.len()].copy_from_slice(bytes);
            arg_bytes[off + bytes.len()] = 0;
            arg_ptrs[i] = arg_bytes.as_ptr() as u64 + off as u64;
            off += bytes.len() + 1;
        }
        arg_ptrs[argv.len()] = 0;
        let ret = self.raw(
            SYS_SPAWN,
            cpath.as_ptr() as u64,
            arg_ptrs.as_ptr() as u64,
            argv.len() as u64,
        );
        Ok(check(ret)? as u32)
    }

    /// `wait(pid)`: block until child `pid` exits, then return its
    /// exit code. Unknown pid: [`Error::BADF`] (matches the kernel
    /// dispatch, which has no other error for this case).
    pub fn wait(&mut self, pid: u32) -> Result<i32> {
        Ok(check(self.raw(SYS_WAIT, pid as u64, 0, 0))? as i32)
    }

    /// `exit(code)`: the raw trap. Returns whatever the kernel
    /// returns — which should be "never", but a mock backend returns
    /// normally, so tests use this to assert on the trap itself.
    pub fn exit_raw(&mut self, code: i32) -> i64 {
        self.raw(SYS_EXIT, code as u64, 0, 0)
    }

    /// `exit(code)`: terminate this process. Never returns; if the
    /// trap ever does return (it must not), spin forever rather than
    /// running on into freed userspace state.
    pub fn exit(&mut self, code: i32) -> ! {
        let _ = self.exit_raw(code);
        loop {
            core::hint::spin_loop();
        }
    }
}

/// Copy `s` into a NUL-terminated stack buffer for a syscall that
/// takes a C string. Errors with [`Error::NAMETOOLONG`] if too long.
fn nul_terminate(s: &str) -> Result<[u8; MAX_PATH_LEN + 1]> {
    let bytes = s.as_bytes();
    if bytes.len() > MAX_PATH_LEN {
        return Err(Error::NAMETOOLONG);
    }
    let mut buf = [0u8; MAX_PATH_LEN + 1];
    buf[..bytes.len()].copy_from_slice(bytes);
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syscall::{
        MockBackend, SYS_CLOSE, SYS_EXIT, SYS_GETPID, SYS_OPEN, SYS_READ, SYS_SPAWN, SYS_UPTIME,
        SYS_WAIT, SYS_WRITE, SYS_YIELD,
    };

    /// Run `f` against a fresh `Syscalls` over a mock with the given
    /// scripted returns, then hand the mock back for assertions.
    /// (`Syscalls` borrows the mock, so it must be dropped before the
    /// mock is inspected — the closure scope does that.)
    fn with_mock<R>(returns: &[i64], f: impl FnOnce(&mut Syscalls) -> R) -> (MockBackend, R) {
        let mut mock = MockBackend::new();
        mock.push_returns(returns);
        let r = {
            let mut sys = Syscalls::new(&mut mock);
            f(&mut sys)
        };
        (mock, r)
    }

    #[test]
    fn write_sends_right_number_and_args() {
        let msg = b"hello";
        let (mock, r) = with_mock(&[5], |sys| sys.write(FD_STDOUT, msg));
        assert_eq!(r, Ok(5));
        let call = mock.last_call().unwrap();
        assert_eq!(call.number, SYS_WRITE);
        assert_eq!(call.a0, FD_STDOUT);
        assert_eq!(call.a1, msg.as_ptr() as u64);
        assert_eq!(call.a2, 5);
        assert_eq!(mock.output, b"hello");
    }

    #[test]
    fn read_sends_right_args_and_maps_return() {
        let mut buf = [0u8; 8];
        let buf_ptr = buf.as_mut_ptr() as u64;
        let (mock, r) = with_mock(&[3], |sys| sys.read(0, &mut buf));
        assert_eq!(r, Ok(3));
        let call = mock.last_call().unwrap();
        assert_eq!((call.number, call.a0, call.a1, call.a2), (SYS_READ, 0, buf_ptr, 8));
    }

    #[test]
    fn error_returns_become_typed_errors() {
        let (mock, r) = with_mock(&[-9, -22, -38, -11], |sys| {
            (
                sys.write(9, b"x"),
                sys.close(1),
                sys.getpid(),
                sys.yield_now(),
            )
        });
        assert_eq!(
            r,
            (
                Err(Error::BADF),
                Err(Error::INVAL),
                Err(Error::NOSYS),
                Err(Error::AGAIN),
            )
        );
        assert_eq!(mock.calls.len(), 4);
    }

    #[test]
    fn open_close_yield_getpid() {
        let (mock, r) = with_mock(&[3, 0, 0, 42], |sys| {
            let opened = sys.open("/etc/passwd", O_RDONLY);
            let closed = sys.close(3);
            let yielded = sys.yield_now();
            let pid = sys.getpid();
            (opened, closed, yielded, pid)
        });
        assert_eq!(r, (Ok(3), Ok(()), Ok(()), Ok(42)));
        assert_eq!(mock.open_paths, vec![b"/etc/passwd".to_vec()]);
        let numbers: Vec<u64> = mock.calls.iter().map(|c| c.number).collect();
        assert_eq!(numbers, vec![SYS_OPEN, SYS_CLOSE, SYS_YIELD, SYS_GETPID]);
        assert_eq!(mock.calls[0].a1, O_RDONLY as u64);
        assert_eq!(mock.calls[1].a0, 3);
    }

    #[test]
    fn uptime_returns_milliseconds_from_syscall_1000() {
        let (mock, r) = with_mock(&[12340], |sys| sys.uptime());
        assert_eq!(r, Ok(12340));
        let call = mock.last_call().unwrap();
        assert_eq!(call.number, SYS_UPTIME);
        assert_eq!((call.a0, call.a1, call.a2), (0, 0, 0));
    }

    #[test]
    fn open_rejects_overlong_path_without_trapping() {
        let long = "a".repeat(MAX_PATH_LEN + 1);
        let (mock, r) = with_mock(&[], |sys| sys.open(&long, O_RDONLY));
        assert_eq!(r, Err(Error::NAMETOOLONG));
        assert!(mock.calls.is_empty());
    }

    #[test]
    fn spawn_sends_path_argv_and_argc() {
        let (mock, r) = with_mock(&[100], |sys| {
            sys.spawn("/bin/hello", &["hello", "a", "b"])
        });
        assert_eq!(r, Ok(100));
        let call = mock.last_call().unwrap();
        assert_eq!(call.number, SYS_SPAWN);
        assert_eq!(call.a2, 3);
        assert_eq!(mock.spawns.len(), 1);
        assert_eq!(mock.spawns[0].0, b"/bin/hello");
        assert_eq!(
            mock.spawns[0].1,
            vec![b"hello".to_vec(), b"a".to_vec(), b"b".to_vec()]
        );
    }

    #[test]
    fn spawn_rejects_too_many_args_without_trapping() {
        let args: [&str; 33] = ["x"; 33];
        let (mock, r) = with_mock(&[], |sys| sys.spawn("/bin/x", &args));
        assert_eq!(r, Err(Error::INVAL));
        assert!(mock.calls.is_empty());
    }

    #[test]
    fn wait_and_exit_raw() {
        let (mock, r) = with_mock(&[7, 0], |sys| (sys.wait(100), sys.exit_raw(3)));
        assert_eq!(r, (Ok(7), 0));
        assert_eq!(mock.calls.len(), 2);
        assert_eq!((mock.calls[0].number, mock.calls[0].a0), (SYS_WAIT, 100));
        assert_eq!((mock.calls[1].number, mock.calls[1].a0), (SYS_EXIT, 3));
    }

    #[test]
    fn wait_error_on_unknown_pid() {
        let (_mock, r) = with_mock(&[-9], |sys| sys.wait(999));
        assert_eq!(r, Err(Error::BADF));
    }
}
