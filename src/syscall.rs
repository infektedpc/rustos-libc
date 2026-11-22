//! Raw syscall layer for RustOS userspace.
//!
//! Syscall numbers and the error convention **must match
//! `rustos-process/src/syscall.rs` exactly** (verified firsthand):
//!
//! | number | name   | signature |
//! |--------|--------|-----------|
//! | 0      | read   | `read(fd, buf, len)` |
//! | 1      | write  | `write(fd, buf, len)` |
//! | 2      | open   | `open(path, flags)` |
//! | 3      | close  | `close(fd)` |
//! | 24     | yield  | `yield()` |
//! | 39     | getpid | `getpid()` |
//! | 57     | spawn  | `spawn(path, argv_ptr, argc)` |
//! | 60     | exit   | `exit(code)` — never returns |
//! | 61     | wait   | `wait(pid)` |
//!
//! The kernel returns a negative errno (`-9`, `-22`, `-38`, …) on
//! error; [`Error::from_raw`] turns that into a typed error and the
//! wrappers in [`crate::io`] never leak a raw negative number.
//!
//! The raw trap is behind the [`SyscallBackend`] trait so host tests
//! can run against [`MockBackend`] instead of executing a real
//! `syscall` instruction on the host. On real RustOS hardware the
//! [`RealBackend`] issues the x86_64 `syscall` instruction
//! (`rax` = number, `rdi`/`rsi`/`rdx` = a0/a1/a2, `rcx`/`r11`
//! clobbered, result in `rax`).

use alloc::collections::VecDeque;
use alloc::vec::Vec;

/// read(fd, buf, len) -> bytes read, or negative errno.
pub const SYS_READ: u64 = 0;
/// write(fd, buf, len) -> bytes written, or negative errno.
pub const SYS_WRITE: u64 = 1;
/// open(path, flags) -> fd, or negative errno.
pub const SYS_OPEN: u64 = 2;
/// close(fd) -> 0, or negative errno.
pub const SYS_CLOSE: u64 = 3;
/// yield() -> 0. Voluntarily give up the CPU.
pub const SYS_YIELD: u64 = 24;
/// getpid() -> pid.
pub const SYS_GETPID: u64 = 39;
/// spawn(path, argv_ptr, argc) -> new pid, or negative errno.
pub const SYS_SPAWN: u64 = 57;
/// exit(code) -> never returns.
pub const SYS_EXIT: u64 = 60;
/// wait(pid) -> exit code, or negative errno; blocks if not exited.
pub const SYS_WAIT: u64 = 61;

/// Standard file descriptors.
pub const FD_STDIN: u64 = 0;
/// Standard file descriptors.
pub const FD_STDOUT: u64 = 1;
/// Standard file descriptors.
pub const FD_STDERR: u64 = 2;

/// Human-readable name for a syscall number, or "unknown".
pub fn syscall_name(n: u64) -> &'static str {
    match n {
        SYS_READ => "read",
        SYS_WRITE => "write",
        SYS_OPEN => "open",
        SYS_CLOSE => "close",
        SYS_YIELD => "yield",
        SYS_GETPID => "getpid",
        SYS_SPAWN => "spawn",
        SYS_EXIT => "exit",
        SYS_WAIT => "wait",
        _ => "unknown",
    }
}

/// All defined syscalls, in number order.
pub const ALL_SYSCALLS: &[(u64, &str)] = &[
    (SYS_READ, "read"),
    (SYS_WRITE, "write"),
    (SYS_OPEN, "open"),
    (SYS_CLOSE, "close"),
    (SYS_YIELD, "yield"),
    (SYS_GETPID, "getpid"),
    (SYS_SPAWN, "spawn"),
    (SYS_EXIT, "exit"),
    (SYS_WAIT, "wait"),
];

/// A kernel error, as a positive errno number (Linux-compatible).
/// The kernel itself returns the negative of this value; see
/// [`Error::from_raw`] and [`Error::as_negative`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Error(pub i64);

impl Error {
    /// ENOENT: no such file or directory.
    pub const NOENT: Error = Error(2);
    /// EINTR: interrupted system call.
    pub const INTR: Error = Error(4);
    /// EIO: I/O error.
    pub const IO: Error = Error(5);
    /// EBADF: bad file descriptor.
    pub const BADF: Error = Error(9);
    /// EAGAIN: try again (would block).
    pub const AGAIN: Error = Error(11);
    /// ENOMEM: out of memory.
    pub const NOMEM: Error = Error(12);
    /// EFAULT: bad address.
    pub const FAULT: Error = Error(14);
    /// EINVAL: invalid argument.
    pub const INVAL: Error = Error(22);
    /// ENOSYS: unknown syscall.
    pub const NOSYS: Error = Error(38);
    /// ECHILD: no child processes.
    pub const CHILD: Error = Error(10);
    /// ENAMETOOLONG: file name too long.
    pub const NAMETOOLONG: Error = Error(36);

    /// The errno number (positive).
    pub fn code(self) -> i64 {
        self.0
    }

    /// The value the kernel actually returns for this error.
    pub fn as_negative(self) -> i64 {
        -self.0
    }

    /// Build from a raw syscall return value: `Some` iff `ret < 0`.
    pub fn from_raw(ret: i64) -> Option<Error> {
        if ret < 0 {
            Some(Error(-ret))
        } else {
            None
        }
    }

    /// Short errno name, or `"EUNKNOWN"` for codes we have no name for.
    pub fn name(self) -> &'static str {
        match self.0 {
            2 => "ENOENT",
            4 => "EINTR",
            5 => "EIO",
            9 => "EBADF",
            10 => "ECHILD",
            11 => "EAGAIN",
            12 => "ENOMEM",
            14 => "EFAULT",
            22 => "EINVAL",
            36 => "ENAMETOOLONG",
            38 => "ENOSYS",
            _ => "EUNKNOWN",
        }
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} ({})", self.name(), self.0)
    }
}

impl core::error::Error for Error {}

/// `Result` with [`Error`] as the error type.
pub type Result<T> = core::result::Result<T, Error>;

/// Turn a raw syscall return into a `Result`: negative means error.
pub fn check(ret: i64) -> Result<i64> {
    match Error::from_raw(ret) {
        Some(e) => Err(e),
        None => Ok(ret),
    }
}

/// Something that can perform one raw syscall. This is the seam
/// that makes the whole crate host-testable: real programs use
/// [`RealBackend`], tests use [`MockBackend`].
pub trait SyscallBackend {
    /// Issue syscall `number` with register arguments `a0`, `a1`,
    /// `a2` and return the raw `rax` value (negative on error).
    fn syscall(&mut self, number: u64, a0: u64, a1: u64, a2: u64) -> i64;
}

/// The real backend: the x86_64 `syscall` instruction.
///
/// `rax` = number, `rdi` = a0, `rsi` = a1, `rdx` = a2; the
/// instruction clobbers `rcx` and `r11`, result in `rax`.
/// On non-x86_64 targets there is no trap yet and every call
/// returns `ERR_NOSYS` (`-38`).
#[derive(Clone, Copy, Debug, Default)]
pub struct RealBackend;

#[cfg(target_arch = "x86_64")]
fn raw_syscall(number: u64, a0: u64, a1: u64, a2: u64) -> i64 {
    let ret: i64;
    unsafe {
        core::arch::asm!(
            "syscall",
            inlateout("rax") number as i64 => ret,
            in("rdi") a0,
            in("rsi") a1,
            in("rdx") a2,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack, preserves_flags),
        );
    }
    ret
}

#[cfg(not(target_arch = "x86_64"))]
fn raw_syscall(_number: u64, _a0: u64, _a1: u64, _a2: u64) -> i64 {
    Error::NOSYS.as_negative()
}

impl SyscallBackend for RealBackend {
    fn syscall(&mut self, number: u64, a0: u64, a1: u64, a2: u64) -> i64 {
        raw_syscall(number, a0, a1, a2)
    }
}

/// One recorded raw syscall: number + three arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyscallCall {
    pub number: u64,
    pub a0: u64,
    pub a1: u64,
    pub a2: u64,
}

impl SyscallCall {
    pub fn name(&self) -> &'static str {
        syscall_name(self.number)
    }
}

/// A mock backend for host tests: records every call, answers from
/// a scripted queue (or `default_return`), and — as a test aid —
/// captures the bytes of `write` calls to fds 1/2 into [`Self::output`],
/// the path of `open` calls into [`Self::open_paths`], and the path
/// and arguments of `spawn` calls into [`Self::spawns`], so tests can
/// assert on what a program passed without dereferencing pointers
/// that died with the caller's stack frame.
pub struct MockBackend {
    /// Every call made, in order.
    pub calls: Vec<SyscallCall>,
    /// Bytes written to stdout/stderr via `SYS_WRITE`.
    pub output: Vec<u8>,
    /// Paths passed to `SYS_OPEN` (without the NUL terminator).
    pub open_paths: Vec<Vec<u8>>,
    /// `(path, argv)` passed to `SYS_SPAWN` (without NULs).
    pub spawns: Vec<(Vec<u8>, Vec<Vec<u8>>)>,
    /// Return value used when the scripted queue is empty.
    pub default_return: i64,
    returns: VecDeque<i64>,
}

impl MockBackend {
    pub fn new() -> Self {
        Self {
            calls: Vec::new(),
            output: Vec::new(),
            open_paths: Vec::new(),
            spawns: Vec::new(),
            default_return: 0,
            returns: VecDeque::new(),
        }
    }

    /// Queue one return value for the next call.
    pub fn push_return(&mut self, value: i64) {
        self.returns.push_back(value);
    }

    /// Queue several return values, in call order.
    pub fn push_returns(&mut self, values: &[i64]) {
        for v in values {
            self.returns.push_back(*v);
        }
    }

    /// The most recent call, if any.
    pub fn last_call(&self) -> Option<SyscallCall> {
        self.calls.last().copied()
    }

    /// All recorded calls with the given syscall number, in order.
    pub fn calls_for(&self, number: u64) -> Vec<SyscallCall> {
        self.calls
            .iter()
            .copied()
            .filter(|c| c.number == number)
            .collect()
    }

    /// Forget recorded calls and captured output (keeps the queue).
    pub fn clear(&mut self) {
        self.calls.clear();
        self.output.clear();
        self.open_paths.clear();
        self.spawns.clear();
    }
}

/// Read a NUL-terminated string at `ptr` (at most `max` bytes),
/// returning the bytes before the NUL. Test-aid only.
fn read_cstr(ptr: u64, max: usize) -> Vec<u8> {
    let mut out = Vec::new();
    if ptr == 0 {
        return out;
    }
    for i in 0..max {
        let b = unsafe { *(ptr as *const u8).add(i) };
        if b == 0 {
            break;
        }
        out.push(b);
    }
    out
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SyscallBackend for MockBackend {
    fn syscall(&mut self, number: u64, a0: u64, a1: u64, a2: u64) -> i64 {
        self.calls.push(SyscallCall {
            number,
            a0,
            a1,
            a2,
        });
        let ret = self.returns.pop_front().unwrap_or(self.default_return);
        // Capture stdout/stderr writes so tests can read what the
        // program printed. `a1`/`a2` are a live pointer/length into
        // the caller's (host test) memory at this point, and only the
        // `ret` bytes the kernel reports as written count — a short
        // write delivers only its first `ret` bytes.
        if number == SYS_WRITE && (a0 == FD_STDOUT || a0 == FD_STDERR) && ret > 0 {
            let n = (ret as usize).min(a2 as usize);
            let bytes = unsafe { core::slice::from_raw_parts(a1 as *const u8, n) };
            self.output.extend_from_slice(bytes);
        }
        if number == SYS_OPEN {
            self.open_paths.push(read_cstr(a0, 256));
        }
        if number == SYS_SPAWN {
            let path = read_cstr(a0, 256);
            let mut args = Vec::new();
            let argc = a2 as usize;
            for i in 0..argc {
                let p = unsafe { *(a1 as *const u64).add(i) };
                args.push(read_cstr(p, 256));
            }
            self.spawns.push((path, args));
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_match_kernel_exactly() {
        assert_eq!(SYS_READ, 0);
        assert_eq!(SYS_WRITE, 1);
        assert_eq!(SYS_OPEN, 2);
        assert_eq!(SYS_CLOSE, 3);
        assert_eq!(SYS_YIELD, 24);
        assert_eq!(SYS_GETPID, 39);
        assert_eq!(SYS_SPAWN, 57);
        assert_eq!(SYS_EXIT, 60);
        assert_eq!(SYS_WAIT, 61);
        assert_eq!(ALL_SYSCALLS.len(), 9);
        let mut prev = 0u64;
        for (i, (n, name)) in ALL_SYSCALLS.iter().enumerate() {
            assert_eq!(syscall_name(*n), *name);
            if i > 0 {
                assert!(*n > prev);
            }
            prev = *n;
        }
        assert_eq!(syscall_name(999), "unknown");
        assert_eq!(FD_STDIN, 0);
        assert_eq!(FD_STDOUT, 1);
        assert_eq!(FD_STDERR, 2);
    }

    #[test]
    fn error_from_raw_and_names() {
        assert_eq!(Error::from_raw(5), None);
        assert_eq!(Error::from_raw(0), None);
        assert_eq!(Error::from_raw(-9), Some(Error::BADF));
        assert_eq!(Error::from_raw(-38), Some(Error::NOSYS));
        assert_eq!(Error::BADF.as_negative(), -9);
        assert_eq!(Error::NOSYS.as_negative(), -38);
        assert_eq!(Error::BADF.name(), "EBADF");
        assert_eq!(Error::INVAL.name(), "EINVAL");
        assert_eq!(Error(1234).name(), "EUNKNOWN");
        assert_eq!(check(-22), Err(Error::INVAL));
        assert_eq!(check(7), Ok(7));
    }

    #[test]
    fn mock_records_calls_and_returns_scripted_values() {
        let mut m = MockBackend::new();
        m.push_returns(&[42, -9]);
        assert_eq!(m.syscall(SYS_GETPID, 0, 0, 0), 42);
        assert_eq!(m.syscall(SYS_CLOSE, 3, 0, 0), -9);
        assert_eq!(m.syscall(SYS_YIELD, 0, 0, 0), 0); // default_return
        assert_eq!(m.calls.len(), 3);
        assert_eq!(m.last_call().unwrap().number, SYS_YIELD);
        assert_eq!(m.calls_for(SYS_GETPID).len(), 1);
        assert_eq!(m.calls_for(SYS_GETPID)[0].a0, 0);
    }

    #[test]
    fn mock_captures_stdout_writes_open_paths_and_spawns() {
        let mut m = MockBackend::new();
        // Report full writes so the whole buffer is "delivered".
        m.default_return = 1024;
        let msg = b"hi";
        m.syscall(SYS_WRITE, FD_STDOUT, msg.as_ptr() as u64, msg.len() as u64);
        m.syscall(SYS_WRITE, FD_STDERR, msg.as_ptr() as u64, msg.len() as u64);
        assert_eq!(m.output, b"hihi");
        let path = b"/bin/hello\0";
        let argv = [b"hello\0".as_ptr() as u64, 0u64];
        m.syscall(
            SYS_SPAWN,
            path.as_ptr() as u64,
            argv.as_ptr() as u64,
            1,
        );
        assert_eq!(m.spawns.len(), 1);
        assert_eq!(m.spawns[0].0, b"/bin/hello");
        assert_eq!(m.spawns[0].1, vec![b"hello".to_vec()]);
        let p2 = b"/etc/passwd\0";
        m.syscall(SYS_OPEN, p2.as_ptr() as u64, 0, 0);
        assert_eq!(m.open_paths, vec![b"/etc/passwd".to_vec()]);
        m.clear();
        assert!(m.calls.is_empty()
            && m.output.is_empty()
            && m.open_paths.is_empty()
            && m.spawns.is_empty());
    }
}
