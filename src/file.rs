//! File I/O wrappers over the syscall layer.
//!
//! There is no `lseek` syscall in RustOS yet, so [`File`] is
//! deliberately seek-less: sequential `read`/`write`/`close` only.
//! Standard descriptors are fds 0/1/2 ([`File::stdin`],
//! [`File::stdout`], [`File::stderr`]).
//!
//! A [`File`] is just a fd — every method takes the [`Syscalls`]
//! it should trap through, so one backend can serve many files and
//! tests can drive files through a [`MockBackend`].

use crate::io::{Syscalls, O_RDONLY};
use crate::syscall::{Result, SyscallBackend, FD_STDERR, FD_STDIN, FD_STDOUT};

/// An open file: just its file descriptor. Methods take the
/// [`Syscalls`] to trap through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct File {
    fd: u64,
}

impl File {
    /// Wrap an already-open fd (e.g. 0/1/2). No syscall is made.
    pub const fn new(fd: u64) -> Self {
        Self { fd }
    }

    /// Standard input (fd 0).
    pub fn stdin() -> Self {
        Self::new(FD_STDIN)
    }

    /// Standard output (fd 1).
    pub fn stdout() -> Self {
        Self::new(FD_STDOUT)
    }

    /// Standard error (fd 2).
    pub fn stderr() -> Self {
        Self::new(FD_STDERR)
    }

    /// The underlying file descriptor.
    pub fn fd(&self) -> u64 {
        self.fd
    }

    /// `open(path, flags)`; returns the open file.
    pub fn open(sys: &mut Syscalls, path: &str, flags: u32) -> Result<Self> {
        Ok(Self::new(sys.open(path, flags)?))
    }

    /// Open a file for reading.
    pub fn open_read(sys: &mut Syscalls, path: &str) -> Result<Self> {
        Self::open(sys, path, O_RDONLY)
    }

    /// Read up to `buf.len()` bytes. Returns bytes read; 0 is EOF.
    pub fn read(&self, sys: &mut Syscalls, buf: &mut [u8]) -> Result<usize> {
        sys.read(self.fd, buf)
    }

    /// Read exactly `buf.len()` bytes unless EOF intervenes.
    /// Returns the number of bytes read, which is `< buf.len()`
    /// only on EOF.
    pub fn read_exact(&self, sys: &mut Syscalls, mut buf: &mut [u8]) -> Result<usize> {
        let mut total = 0;
        while !buf.is_empty() {
            match self.read(sys, buf)? {
                0 => break, // EOF
                n => {
                    // Clamp: a backend must never report more than
                    // was asked for, but do not panic if one does.
                    let n = n.min(buf.len());
                    total += n;
                    let tmp = core::mem::take(&mut buf);
                    buf = &mut tmp[n..];
                }
            }
        }
        Ok(total)
    }

    /// Write bytes. Returns bytes written (may be short).
    pub fn write(&self, sys: &mut Syscalls, buf: &[u8]) -> Result<usize> {
        sys.write(self.fd, buf)
    }

    /// Write the whole buffer, looping over short writes.
    pub fn write_all(&self, sys: &mut Syscalls, mut buf: &[u8]) -> Result<()> {
        while !buf.is_empty() {
            match self.write(sys, buf)? {
                0 => break, // cannot make progress; treat as done
                // Clamp: a backend must never report more than was
                // asked for, but do not panic if one does.
                n => buf = &buf[n.min(buf.len())..],
            }
        }
        Ok(())
    }

    /// Write a whole string.
    pub fn write_str(&self, sys: &mut Syscalls, s: &str) -> Result<()> {
        self.write_all(sys, s.as_bytes())
    }

    /// Write a string followed by `\n`.
    pub fn write_line(&self, sys: &mut Syscalls, s: &str) -> Result<()> {
        self.write_all(sys, s.as_bytes())?;
        self.write_all(sys, b"\n")
    }

    /// Close the file, consuming it.
    pub fn close(self, sys: &mut Syscalls) -> Result<()> {
        sys.close(self.fd)
    }
}

/// A `core::fmt::Write` sink that writes to one fd through a
/// backend — the bridge between `write!`/`core::write_fmt` and the
/// `write` syscall:
///
/// ```ignore
/// use core::fmt::Write as _;
/// let mut out = FileWriter::new(1, sys.backend());
/// write!(out, "pid={}", pid).unwrap();
/// ```
pub struct FileWriter<'a> {
    fd: u64,
    backend: &'a mut dyn SyscallBackend,
}

impl<'a> FileWriter<'a> {
    pub fn new(fd: u64, backend: &'a mut dyn SyscallBackend) -> Self {
        Self { fd, backend }
    }

    /// A writer for stdout (fd 1).
    pub fn stdout(backend: &'a mut dyn SyscallBackend) -> Self {
        Self::new(FD_STDOUT, backend)
    }

    /// A writer for stderr (fd 2).
    pub fn stderr(backend: &'a mut dyn SyscallBackend) -> Self {
        Self::new(FD_STDERR, backend)
    }
}

impl core::fmt::Write for FileWriter<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let mut sys = Syscalls::new(&mut *self.backend);
        File::new(self.fd)
            .write_all(&mut sys, s.as_bytes())
            .map_err(|_| core::fmt::Error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{fmt::format_printf, pargs};
    use crate::syscall::{MockBackend, FD_STDOUT, SYS_CLOSE, SYS_OPEN, SYS_READ, SYS_WRITE};
    use core::fmt::Write as _;

    #[test]
    fn file_open_read_write_close_against_mock() {
        let mut mock = MockBackend::new();
        mock.push_returns(&[3, 5, 0]); // open -> fd 3, write -> 5, close -> 0
        let mut sys = Syscalls::new(&mut mock);
        let f = File::open(&mut sys, "/tmp/x", O_RDONLY).unwrap();
        assert_eq!(f.fd(), 3);
        assert_eq!(f.write(&mut sys, b"hello").unwrap(), 5);
        assert_eq!(f.close(&mut sys), Ok(()));
        drop(sys);
        assert_eq!(mock.open_paths, vec![b"/tmp/x".to_vec()]);
        let numbers: Vec<u64> = mock.calls.iter().map(|c| c.number).collect();
        assert_eq!(numbers, vec![SYS_OPEN, SYS_WRITE, SYS_CLOSE]);
        assert_eq!(mock.output, b""); // wrote to fd 3, not stdout
    }

    #[test]
    fn write_all_loops_over_short_writes() {
        let mut mock = MockBackend::new();
        mock.push_returns(&[2, 2, 1]); // three short writes
        let mut sys = Syscalls::new(&mut mock);
        let f = File::stdout();
        assert_eq!(f.write_all(&mut sys, b"hello"), Ok(()));
        drop(sys);
        assert_eq!(mock.calls_for(SYS_WRITE).len(), 3);
        assert_eq!(mock.output, b"hello");
    }

    #[test]
    fn read_exact_stops_at_eof() {
        let mut mock = MockBackend::new();
        mock.push_returns(&[3, 0]); // 3 bytes, then EOF
        let mut sys = Syscalls::new(&mut mock);
        let f = File::stdin();
        let mut buf = [0u8; 8];
        assert_eq!(f.read_exact(&mut sys, &mut buf).unwrap(), 3);
        drop(sys);
        assert_eq!(mock.calls_for(SYS_READ).len(), 2);
    }

    #[test]
    fn file_error_propagates() {
        let mut mock = MockBackend::new();
        mock.push_returns(&[-9]); // open -> EBADF
        let mut sys = Syscalls::new(&mut mock);
        assert_eq!(
            File::open(&mut sys, "/nope", O_RDONLY),
            Err(crate::syscall::Error::BADF)
        );
    }

    #[test]
    fn fmt_write_goes_to_stdout() {
        let mut mock = MockBackend::new();
        // FileWriter writes via write_all; the mock must answer each
        // write with the requested length. Queue generous returns.
        mock.push_returns(&[64, 64, 64, 64, 64, 64, 64, 64]);
        let mut out = FileWriter::stdout(&mut mock);
        let msg = format_printf("pid=%d", pargs![42]);
        write!(out, "{msg}").unwrap();
        drop(out);
        assert_eq!(mock.output, b"pid=42");
        assert_eq!(mock.calls_for(SYS_WRITE).len(), 1);
        assert_eq!(mock.calls_for(SYS_WRITE)[0].a0, FD_STDOUT);
    }

    #[test]
    fn stdio_fds() {
        assert_eq!(File::stdin().fd(), 0);
        assert_eq!(File::stdout().fd(), 1);
        assert_eq!(File::stderr().fd(), 2);
    }
}
