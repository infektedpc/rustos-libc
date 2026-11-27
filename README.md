# rustos-libc

`no_std` userspace runtime (libc) for RustOS. Core + alloc only, no
external crates. Host-testable: every syscall goes through a
replaceable backend, so the test suite runs against a mock and never
executes a real syscall on the host.

## Syscalls

Numbers match `rustos-process/src/syscall.rs` exactly (verified
firsthand) — do not change one side without the other:

| number | name | signature |
|--------|------|-----------|
| 0 | read | `read(fd, buf, len)` |
| 1 | write | `write(fd, buf, len)` |
| 2 | open | `open(path, flags)` |
| 3 | close | `close(fd)` |
| 24 | yield | `yield()` |
| 39 | getpid | `getpid()` |
| 57 | spawn | `spawn(path, argv_ptr, argc)` |
| 60 | exit | `exit(code)` — never returns |
| 61 | wait | `wait(pid)` |

Errors are negative errnos on the wire (`-9` EBADF, `-22` EINVAL,
`-38` ENOSYS, …); the wrappers translate them into `Error` (a
positive errno code with a name), so callers never see a raw
negative number.

## API

| Module | What it provides |
|---|---|
| `syscall` | Number constants, `Error`/`Result`/`check`, `SyscallBackend` trait, `RealBackend` (x86_64 `syscall` instruction: `rax`=number, `rdi`/`rsi`/`rdx`=args — the default backend since Batch 11 wired the kernel's MSR entry), `Int80Backend` (the `int 0x80` trap — the fallback, selected by the `int80-backend` feature), `default_backend()` (the constructor real programs call), `MockBackend` (records calls, scripted returns, captures stdout/stderr writes, open paths, spawn path+argv) |
| `io` | `Syscalls`: typed wrappers `read`/`write`/`write_str`/`print`/`open`/`close`/`yield_now`/`getpid`/`spawn`/`wait`/`exit` (+ `exit_raw` for tests). `open`/`spawn` NUL-terminate paths on a stack buffer; limits: path ≤ 255 bytes, spawn ≤ 32 args / ≤ 1024 bytes of arg text |
| `heap` | `Heap`: first-fit free-list allocator over a caller-provided region — splits on alloc, coalesces neighbours on free, double-free and invalid-free detection, `alloc`/`free`/`realloc` with alignment. `RustosAlloc`: `GlobalAlloc` over one heap behind a spinlock; `ALLOCATOR` is the process-wide instance, installed with `init_heap(base, size)` |
| `fmt` | `printf`-style formatting backed by `core::fmt`: `format_printf(fmt, args)`, `printf_to(writer, fmt, args)`, `pargs![...]`. Conversions `%d %i %u %x %X %s %c %%`, width, `%-` left-align, `%0` zero-pad. Unknown conversions and missing args are written literally, never dropped |
| `file` | `File`: seek-less file I/O (there is no `lseek` syscall yet) — `open`/`open_read`/`read`/`read_exact`/`write`/`write_all`/`write_str`/`write_line`/`close`, plus `stdin()`/`stdout()`/`stderr()`. `FileWriter`: a `core::fmt::Write` sink for one fd, so `write!(out, "pid={}", pid)` works |
| `start` | `StartInfo` (heap base + size), `init_runtime`, `start(info, main)`: install the heap, run `main`, `exit` with its code |
| macros | `uprint!(&mut sys, ...)` / `uprintln!(&mut sys, ...)` — formatted output to stdout |

Convenience re-exports at the crate root: `Vec`, `String`, `Box`,
`format!` (from `alloc`), so a program needs only this crate for
the usual heap types.

## How a userspace program uses this crate

```rust
#![no_std]
#![no_main]

use rustos_libc::{start, StartInfo, Syscalls, RealBackend};
use core::panic::PanicInfo;

// The global allocator is rustos-libc's ALLOCATOR (it carries
// #[global_allocator] on freestanding targets when the
// `global-allocator` feature is enabled — userspace programs enable
// it in their Cargo.toml) — a program must NOT declare a second one
// of its own: `start()` installs the heap into ALLOCATOR, and a
// second, never-initialised allocator would fail every allocation
// (Batch 9: both live programs exited 1 before printing anything).

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop { core::hint::spin_loop(); } // or print-and-exit; your choice
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    // The heap region comes from the kernel (see "Known limits":
    // the handoff ABI is not defined yet — obtain base/size by
    // whatever the kernel of the day provides).
    let heap_base = 0x7000_0000 as *mut u8; // placeholder
    let heap_size = 0x2000;                 // placeholder
    start(StartInfo::new(heap_base, heap_size), main)
}

fn main() -> i32 {
    let mut backend = RealBackend;
    let mut sys = Syscalls::new(&mut backend);
    let _ = sys.print("Hello, RustOS!\n");
    0
}
```

`examples/hello.rs` is a host-runnable version of this shape: its
`run(&mut Syscalls)` holds the program logic (one `write`, no
spawn), driven in tests through `MockBackend`. On a Linux x86_64
host the example also runs for real, because Linux shares syscall
numbers 1 (`write`) and 60 (`exit`) with RustOS.

## Tests

```sh
cargo test            # all host tests, mock backend only
cargo build --examples  # examples/hello.rs compiles
```

## Known limits — what is NOT verified

- **No live boot.** QEMU is not installed on this VM; nothing here
  has run against the real RustOS kernel. No real syscall round-trip
  has happened yet: the numbers and argument order match
  `rustos-process`'s dispatch by reading its source, not by
  executing against it.
- **Heap handoff ABI undefined.** The kernel does not yet define how
  a program learns its heap region (registers / fixed struct /
  initial stack). `start` is the receiving end of that contract;
  the kernel side is still to be defined, and the placeholder
  addresses in the snippet above are not real.
- **`spawn` argv convention is a proposal.** The kernel dispatch
  currently ignores the path/argv pointers and just allocates a
  pid, so the `spawn(path, argv_ptr, argc)` layout this crate sends
  is what the real handler *should* read — it has not been confirmed
  against a handler implementation.
- **Heap is one fixed region only.** No growing the heap (no
  `sbrk`/`mmap` syscall exists yet); allocation fails when the
  region is full. The allocator's spinlock assumes a single CPU —
  no SMP yet, and never call it from an interrupt handler while
  holding the lock.
- **No `lseek`, no signals, no environment, no argv parsing, no
  TLS, no `fork`** (RustOS has no COW yet), no `atexit`. `wait`
  reports the kernel's `EBADF` for an unknown pid because that is
  what the dispatch returns. `read` on stdin returns EOF (0) in the
  kernel's fake state — keyboard input is not wired to `read` yet.
- **Paths are capped** at 255 bytes and `spawn` at 32 args /
  1024 bytes of argument text (stack buffers); longer inputs are
  rejected before trapping.
