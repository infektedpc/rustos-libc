//! Heap allocator over a caller-provided memory region.
//!
//! A userspace program has no `sbrk`/`mmap` (the kernel hands it one
//! heap region at startup — see [`crate::start`]); [`Heap`] manages
//! that region with a first-fit free list, splitting blocks on
//! allocation and coalescing neighbours on free, so freed memory is
//! reused instead of leaking away like a bump allocator.
//!
//! Layout of one block (16-byte header):
//!
//! ```text
//! +------------+-----------------+----------------------+
//! | size (u64) | next / self (u64)| payload ...          |
//! +------------+-----------------+----------------------+
//! ```
//!
//! * `size` is the whole block size including the header, always a
//!   multiple of 16.
//! * Free blocks use the second word as the next-free-list link.
//! * Used blocks use the second word as a back-pointer: the payload
//!   pointer returned to the caller always has the block's own
//!   address stored 8 bytes before it, so `free`/`realloc` can find
//!   the header without a side table. (When the payload is pushed
//!   forward for alignment, the back-pointer sits in the slack
//!   between header and payload.)
//!
//! [`RustosAlloc`] wraps one [`Heap`] behind a spinlock and
//! implements [`core::alloc::GlobalAlloc`], so a program can do
//! `#[global_allocator] static ALLOC: RustosAlloc = RustosAlloc::new();`
//! and then use `alloc::vec::Vec`, `String`, `Box`, …
//!
//! Single-core assumption: the spinlock only serialises callers on
//! one CPU. There is no SMP yet, and a program that calls the
//! allocator from an interrupt handler while holding the lock will
//! deadlock — same rule as every other RustOS spinlock.

use core::alloc::{GlobalAlloc, Layout};
use core::ptr::{self};
use core::sync::atomic::{AtomicBool, Ordering};

/// Block header size in bytes (two `u64` words).
const HEADER_SIZE: usize = 16;
/// Smallest block the allocator will ever create or keep.
const MIN_BLOCK: usize = 16;
/// Smallest alignment the allocator guarantees.
const MIN_ALIGN: usize = 16;

fn align_up(v: usize, align: usize) -> usize {
    (v + align - 1) & !(align - 1)
}

/// One free block. Its in-memory layout is exactly the block header
/// described above: `size` then `next`.
#[repr(C)]
struct FreeBlock {
    size: usize,
    next: *mut FreeBlock,
}

/// A heap over a fixed region supplied by the caller.
pub struct Heap {
    base: *mut u8,
    size: usize,
    free_list: *mut FreeBlock,
}

impl Heap {
    /// Create a heap over `[base, base + size)`. The region edges
    /// are rounded in to a 16-byte boundary (wasting at most 30
    /// bytes); a region too small to hold one block yields an empty
    /// heap that refuses every allocation.
    ///
    /// # Safety
    ///
    /// `base` must point to `size` bytes of writable memory that
    /// stays alive and is owned by this heap for as long as it is
    /// used, and must not overlap any other live heap.
    pub unsafe fn new(base: *mut u8, size: usize) -> Self {
        let start = align_up(base as usize, MIN_ALIGN);
        let end = (base as usize + size) & !(MIN_ALIGN - 1);
        let mut heap = Heap {
            base: start as *mut u8,
            size: end.saturating_sub(start),
            free_list: ptr::null_mut(),
        };
        if heap.size >= MIN_BLOCK {
            let block = heap.base as *mut FreeBlock;
            ptr::write(
                block,
                FreeBlock {
                    size: heap.size,
                    next: ptr::null_mut(),
                },
            );
            heap.free_list = block;
        }
        heap
    }

    /// Total usable bytes in the region (after edge rounding).
    pub fn total_size(&self) -> usize {
        self.size
    }

    /// Bytes currently free (sum of the free list).
    pub fn free_bytes(&self) -> usize {
        let mut total = 0;
        let mut cur = self.free_list;
        while !cur.is_null() {
            total += unsafe { (*cur).size };
            cur = unsafe { (*cur).next };
        }
        total
    }

    /// Bytes currently in use (headers included).
    pub fn used_bytes(&self) -> usize {
        self.size - self.free_bytes()
    }

    /// Allocate `size` bytes with `align` alignment.
    /// Returns `None` if `align` is not a power of two or the heap
    /// cannot satisfy the request. A `size` of 0 allocates a minimal
    /// block (never returns a dangling pointer).
    pub fn alloc(&mut self, size: usize, align: usize) -> Option<*mut u8> {
        if !align.is_power_of_two() {
            return None;
        }
        let align = align.max(MIN_ALIGN);
        let payload = align_up(size.max(1), MIN_ALIGN);
        let need = payload
            .checked_add(HEADER_SIZE)?
            .checked_add(align - MIN_ALIGN)?
            .max(MIN_BLOCK);
        // First-fit walk.
        let mut prev: *mut FreeBlock = ptr::null_mut();
        let mut cur = self.free_list;
        while !cur.is_null() {
            let block_size = unsafe { (*cur).size };
            if block_size >= need {
                let next = unsafe { (*cur).next };
                if prev.is_null() {
                    self.free_list = next;
                } else {
                    unsafe { (*prev).next = next };
                }
                let used_size = if block_size - need >= MIN_BLOCK {
                    // Split: the tail becomes a new free block.
                    let rem = unsafe { (cur as *mut u8).add(need) } as *mut FreeBlock;
                    unsafe {
                        ptr::write(
                            rem,
                            FreeBlock {
                                size: block_size - need,
                                next: self.free_list,
                            },
                        )
                    };
                    self.free_list = rem;
                    need
                } else {
                    block_size
                };
                let block_addr = cur as usize;
                let payload_ptr =
                    align_up(block_addr + HEADER_SIZE, align) as *mut u8;
                unsafe {
                    // Header: total size, then the back-pointer that
                    // `free`/`realloc` read from `payload - 8`.
                    ptr::write(cur as *mut usize, used_size);
                    ptr::write(
                        (payload_ptr as *mut u64).sub(1),
                        block_addr as u64,
                    );
                }
                return Some(payload_ptr);
            }
            prev = cur;
            cur = unsafe { (*cur).next };
        }
        None
    }

    /// Free a pointer previously returned by [`Heap::alloc`] or
    /// [`Heap::realloc`]. Returns `false` (and does nothing) for a
    /// null pointer, a pointer outside this heap, a corrupted header,
    /// or a double free — freeing the same pointer twice is detected
    /// by scanning the free list.
    pub fn free(&mut self, ptr: *mut u8) -> bool {
        let Some(block) = self.block_for(ptr) else {
            return false;
        };
        // Double free: the block is already on the free list.
        let mut cur = self.free_list;
        while !cur.is_null() {
            if cur == block {
                return false;
            }
            cur = unsafe { (*cur).next };
        }
        let block_addr = block as usize;
        let block_size = unsafe { (*block).size };
        self.insert_free(block_addr, block_size);
        true
    }

    /// Resize an allocation. `ptr` may be null (acts like `alloc`).
    /// `new_size == 0` frees and returns `None`. Grows by allocating
    /// a new block, copying the old contents, and freeing the old
    /// block; shrinks in place when the tail left over is big enough
    /// to become a free block. Returns `None` (old block untouched)
    /// if a grow cannot be satisfied.
    pub fn realloc(&mut self, ptr: *mut u8, new_size: usize, align: usize) -> Option<*mut u8> {
        if ptr.is_null() {
            return self.alloc(new_size, align);
        }
        if new_size == 0 {
            self.free(ptr);
            return None;
        }
        let Some(block) = self.block_for(ptr) else {
            return None;
        };
        let block_addr = block as usize;
        let block_size = unsafe { (*block).size };
        let old_capacity = block_size - (ptr as usize - block_addr);
        let new_payload = align_up(new_size, MIN_ALIGN);
        if new_payload <= old_capacity {
            // Fits: shrink in place, freeing the tail if worthwhile.
            let used = (ptr as usize - block_addr) + new_payload;
            let tail = block_size - used;
            if tail >= MIN_BLOCK {
                unsafe { ptr::write(block as *mut usize, used) };
                let rem = unsafe { (block as *mut u8).add(used) } as *mut FreeBlock;
                unsafe {
                    ptr::write(
                        rem,
                        FreeBlock {
                            size: tail,
                            next: self.free_list,
                        },
                    )
                };
                self.free_list = rem;
            }
            return Some(ptr);
        }
        let new_ptr = self.alloc(new_size, align)?;
        unsafe {
            ptr::copy_nonoverlapping(ptr, new_ptr, old_capacity.min(new_size));
        }
        self.free(ptr);
        Some(new_ptr)
    }

    /// Validate `ptr` and return its block header, or `None`.
    fn block_for(&self, ptr: *mut u8) -> Option<*mut FreeBlock> {
        if ptr.is_null() {
            return None;
        }
        // The back-pointer sits 8 bytes before the payload. Read it
        // unaligned: callers may hand us a deliberately bad pointer
        // (e.g. payload + 1) and this must return `false`, not trap.
        let block_addr = unsafe { ptr::read_unaligned((ptr as *const u64).sub(1)) } as usize;
        if block_addr % MIN_ALIGN != 0 {
            return None;
        }
        let base = self.base as usize;
        if block_addr < base || block_addr >= base + self.size {
            return None;
        }
        let block = block_addr as *mut FreeBlock;
        let block_size = unsafe { (*block).size };
        if block_size < MIN_BLOCK
            || block_size % MIN_ALIGN != 0
            || block_addr + block_size > base + self.size
        {
            return None;
        }
        let p = ptr as usize;
        if p < block_addr + HEADER_SIZE || p >= block_addr + block_size {
            return None;
        }
        Some(block)
    }

    /// Insert the block `[addr, addr + size)` into the free list in
    /// address order, coalescing with adjacent free neighbours.
    fn insert_free(&mut self, addr: usize, size: usize) {
        let mut prev: *mut FreeBlock = ptr::null_mut();
        let mut cur = self.free_list;
        while !cur.is_null() && (cur as usize) < addr {
            prev = cur;
            cur = unsafe { (*cur).next };
        }
        let mut merged_size = size;
        // Coalesce with the next block.
        if !cur.is_null() && addr + size == cur as usize {
            merged_size += unsafe { (*cur).size };
            cur = unsafe { (*cur).next };
        }
        // Coalesce with the previous block.
        if !prev.is_null() && (prev as usize) + unsafe { (*prev).size } == addr {
            unsafe { (*prev).size += merged_size };
            unsafe { (*prev).next = cur };
            return;
        }
        let merged = addr as *mut FreeBlock;
        unsafe {
            ptr::write(
                merged,
                FreeBlock {
                    size: merged_size,
                    next: cur,
                },
            )
        };
        if prev.is_null() {
            self.free_list = merged;
        } else {
            unsafe { (*prev).next = merged };
        }
    }
}

/// A [`GlobalAlloc`] over one heap, behind a spinlock.
///
/// Create one as a `static` and initialise it once with
/// [`init_heap`] (or [`RustosAlloc::init`]) before any allocation.
/// The process-wide instance is [`ALLOCATOR`]: on a freestanding
/// target with the `global-allocator` feature it is also the
/// program's global allocator (see its docs), so a program uses it
/// by calling [`init_heap`] — it must NOT declare a second
/// `#[global_allocator]` static of its own (see [`ALLOCATOR`]: an
/// uninitialised second instance fails every allocation).
pub struct RustosAlloc {
    heap: core::cell::UnsafeCell<Option<Heap>>,
    locked: AtomicBool,
}

// The heap lives in a `static` and is only touched under the lock.
unsafe impl Sync for RustosAlloc {}

impl RustosAlloc {
    pub const fn new() -> Self {
        Self {
            heap: core::cell::UnsafeCell::new(None),
            locked: AtomicBool::new(false),
        }
    }

    /// Initialise the global heap over `[base, base + size)`.
    /// Returns `true` if this call initialised it, `false` if it
    /// was already initialised (the existing heap is kept).
    ///
    /// # Safety
    ///
    /// Same requirements as [`Heap::new`]; call once, before any
    /// allocation through this allocator.
    pub unsafe fn init(&self, base: *mut u8, size: usize) -> bool {
        let _guard = self.lock();
        let slot = &mut *self.heap.get();
        if slot.is_some() {
            return false;
        }
        *slot = Some(Heap::new(base, size));
        true
    }

    /// True once [`RustosAlloc::init`] has run.
    pub fn is_initialized(&self) -> bool {
        let _guard = self.lock();
        unsafe { (*self.heap.get()).is_some() }
    }

    /// Bytes currently free in the global heap (0 if uninitialised).
    pub fn free_bytes(&self) -> usize {
        let _guard = self.lock();
        unsafe { (*self.heap.get()).as_ref().map_or(0, Heap::free_bytes) }
    }

    fn lock(&self) -> SpinGuard<'_> {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
        SpinGuard { alloc: self }
    }
}

struct SpinGuard<'a> {
    alloc: &'a RustosAlloc,
}

impl Drop for SpinGuard<'_> {
    fn drop(&mut self) {
        self.alloc.locked.store(false, Ordering::Release);
    }
}

unsafe impl GlobalAlloc for RustosAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _guard = self.lock();
        let Some(heap) = (unsafe { &mut *self.heap.get() }) else {
            return ptr::null_mut();
        };
        heap.alloc(layout.size(), layout.align())
            .unwrap_or(ptr::null_mut())
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        let _guard = self.lock();
        if let Some(heap) = unsafe { &mut *self.heap.get() } {
            heap.free(ptr);
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _guard = self.lock();
        let Some(heap) = (unsafe { &mut *self.heap.get() }) else {
            return ptr::null_mut();
        };
        heap.realloc(ptr, new_size, layout.align())
            .unwrap_or(ptr::null_mut())
    }
}

/// The process-wide allocator. Initialise it once with
/// [`init_heap`] (see [`crate::start`]) before allocating.
///
/// This is the ONE allocator instance in a userspace program —
/// and, on a freestanding target (`target_os = "none"`) with the
/// `global-allocator` feature enabled, it is also the program's
/// global allocator (see the attribute below), so
/// `alloc::format!`/`Vec`/`String` allocate through the heap
/// `start()` installs. A program must NOT declare its own
/// `#[global_allocator] static ALLOC`: that would be a second,
/// never-initialised `RustosAlloc`, and the first allocation
/// through it would fail and panic — which is exactly how the
/// Batch 9 live boot died (both programs exited 1 before printing
/// anything; `start()` had initialised this `ALLOCATOR` while
/// `format!` allocated through the programs' own dead `ALLOC`).
/// The attribute is target- AND feature-gated, not `cfg(test)`:
/// host test binaries (target `linux`) keep the standard
/// allocator, so unit tests are unaffected; and the kernel — which
/// links this crate but has its own global allocator — never
/// enables the feature, so the two never collide in one binary.
/// Userspace programs enable it in their `Cargo.toml`
/// (`rustos-libc = { features = ["global-allocator"] }`).
#[cfg_attr(all(target_os = "none", feature = "global-allocator"), global_allocator)]
pub static ALLOCATOR: RustosAlloc = RustosAlloc::new();

/// Initialise the process-wide heap over `[base, base + size)`.
/// Returns `true` if this call initialised it, `false` if the heap
/// was already initialised. See [`RustosAlloc::init`].
///
/// # Safety
///
/// Same requirements as [`Heap::new`]; call once, before any
/// allocation.
pub unsafe fn init_heap(base: *mut u8, size: usize) -> bool {
    ALLOCATOR.init(base, size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// A 16-aligned stack buffer for heap tests.
    #[repr(align(16))]
    struct AlignedBuf([u8; 2048]);

    /// A heap over a boxed 16-aligned buffer. The buffer is boxed
    /// (not a stack local) so moving the pair cannot invalidate the
    /// raw pointer the `Heap` holds into it.
    fn heap() -> (alloc::boxed::Box<AlignedBuf>, Heap) {
        let mut buf = alloc::boxed::Box::new(AlignedBuf([0u8; 2048]));
        let h = unsafe { Heap::new(buf.0.as_mut_ptr(), buf.0.len()) };
        (buf, h)
    }

    #[test]
    fn alloc_free_reuse() {
        let (_buf, mut h) = heap();
        let total = h.total_size();
        let a = h.alloc(64, 8).unwrap();
        unsafe { ptr::write_bytes(a, 0xAB, 64) };
        assert!(h.free(a));
        assert_eq!(h.free_bytes(), total);
        // Freed memory is reused: same size comes back at the same spot.
        let b = h.alloc(64, 8).unwrap();
        assert_eq!(a, b);
        assert_eq!(unsafe { *b }, 0xAB); // old contents still there
        assert!(h.free(b));
    }

    #[test]
    fn alignment_is_honoured() {
        let (_buf, mut h) = heap();
        for align in [1, 2, 8, 16, 32, 64, 128, 256] {
            let p = h.alloc(13, align).unwrap();
            assert_eq!(p as usize % align.max(16), 0, "align {align}");
            assert!(h.free(p));
        }
        assert_eq!(h.alloc(10, 3), None); // not a power of two
        assert_eq!(h.alloc(10, 0), None);
    }

    #[test]
    fn out_of_memory() {
        let (_buf, mut h) = heap();
        assert_eq!(h.alloc(h.total_size() + 1, 16), None);
        assert_eq!(h.alloc(h.total_size(), 16), None); // header does not fit
        let big = h.alloc(h.total_size() - HEADER_SIZE, 16).unwrap();
        assert_eq!(h.alloc(1, 16), None);
        assert!(h.free(big));
        assert!(h.alloc(1, 16).is_some());
    }

    #[test]
    fn double_free_detected_and_harmless() {
        let (_buf, mut h) = heap();
        let total = h.total_size();
        let a = h.alloc(32, 16).unwrap();
        assert!(h.free(a));
        let free_after_first = h.free_bytes();
        assert!(!h.free(a)); // second free refused
        assert_eq!(h.free_bytes(), free_after_first);
        // The heap is still usable and the block is still free.
        let b = h.alloc(32, 16).unwrap();
        assert_eq!(a, b);
        assert!(h.free(b));
        assert_eq!(h.free_bytes(), total);
    }

    #[test]
    fn invalid_free_rejected() {
        let (_buf, mut h) = heap();
        assert!(!h.free(ptr::null_mut()));
        let mut stack_byte = 0u8;
        assert!(!h.free(&mut stack_byte as *mut u8));
        let p = h.alloc(16, 16).unwrap();
        // One byte into the payload: back-pointer reads garbage.
        assert!(!h.free(unsafe { p.add(1) }));
        assert!(h.free(p));
    }

    #[test]
    fn realloc_grows_preserving_data_and_shrinks() {
        let (_buf, mut h) = heap();
        let p = h.alloc(16, 16).unwrap();
        for i in 0..16 {
            unsafe { *p.add(i) = i as u8 };
        }
        let q = h.realloc(p, 200, 16).unwrap();
        for i in 0..16 {
            assert_eq!(unsafe { *q.add(i) }, i as u8);
        }
        // Shrinking keeps the data and the pointer.
        let r = h.realloc(q, 8, 16).unwrap();
        assert_eq!(r, q);
        for i in 0..8 {
            assert_eq!(unsafe { *r.add(i) }, i as u8);
        }
        assert!(h.free(r));
        // realloc(null) is alloc; realloc(_, 0) is free.
        let s = h.realloc(ptr::null_mut(), 32, 16).unwrap();
        assert_eq!(h.realloc(s, 0, 16), None);
    }

    #[test]
    fn coalescing_reclaims_adjacent_blocks() {
        let (_buf, mut h) = heap();
        let total = h.total_size();
        let a = h.alloc(64, 16).unwrap();
        let b = h.alloc(64, 16).unwrap();
        let c = h.alloc(64, 16).unwrap();
        assert!(h.free(a));
        assert!(h.free(b)); // adjacent to a: coalesces
        assert!(h.free(c));
        assert_eq!(h.free_bytes(), total);
        // The whole heap is one block again.
        assert!(h.alloc(total - HEADER_SIZE, 16).is_some());
    }

    #[test]
    fn many_allocs_and_vec_string_work() {
        let (_buf, mut h) = heap();
        let mut ptrs = Vec::new();
        for _ in 0..32 {
            ptrs.push(h.alloc(48, 16).unwrap());
        }
        for p in &ptrs {
            assert!(h.free(*p));
        }
        assert_eq!(h.free_bytes(), h.total_size());
        // The same heap can back alloc collections via GlobalAlloc
        // once it is installed (tested below through ALLOCATOR).
    }

    #[test]
    fn global_allocator_smoke() {
        // One test owns ALLOCATOR (tests share a process). Allocate
        // a 16-aligned host buffer and initialise once.
        let layout = Layout::from_size_align(4096, 16).unwrap();
        let buf = unsafe { std::alloc::alloc(layout) };
        assert!(!buf.is_null());
        let first = unsafe { init_heap(buf, 4096) };
        // If another run already initialised it (re-run in one
        // process), init returns false but the heap is still usable.
        let _ = first;
        assert!(ALLOCATOR.is_initialized());
        let free_before = ALLOCATOR.free_bytes();
        assert!(free_before > 0);
        let layout64 = Layout::from_size_align(64, 8).unwrap();
        let p = unsafe { ALLOCATOR.alloc(layout64) };
        assert!(!p.is_null());
        assert_eq!(p as usize % 8, 0);
        assert!(ALLOCATOR.free_bytes() < free_before);
        unsafe { ALLOCATOR.dealloc(p, layout64) };
        assert_eq!(ALLOCATOR.free_bytes(), free_before);
        // Vec<String> through the global allocator.
        let mut v: Vec<alloc::string::String> = Vec::new();
        // NOTE: this Vec uses the *test binary's* global allocator,
        // not ALLOCATOR — it only proves the types exist. On a
        // freestanding target with the `global-allocator` feature
        // ALLOCATOR *is* the global allocator (see its docs), so
        // this is the one place the two differ: host tests allocate
        // through std's allocator, live programs through ALLOCATOR.
        // The orchestrator's allocator-wiring check (rustos-build)
        // pins that the shipped program binaries contain exactly
        // one `RustosAlloc` static.
        v.push(alloc::string::String::from("hello"));
        assert_eq!(v[0], "hello");
        // Deliberately leak the allocation: ALLOCATOR still points
        // at it, and other tests in this process read through it.
        // (The pointer itself is `Copy`, so there is nothing to
        // `forget` — just do not call `dealloc`.)
    }
}
