"""Physical KV-cache block allocation."""

from __future__ import annotations

from collections import deque

from nimbus.sequence import Sequence


class OutOfBlocks(RuntimeError):
    pass


class BlockAllocator:
    """Free list of physical blocks with reference counts.

    Invariant: a block is either on the free list (refcount 0) or owned by
    at least one block table (refcount >= 1), never both. Reference counts
    above 1 let speculative decoding share a prefix without copying.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free = deque(range(num_blocks))
        self.refcount = [0] * num_blocks

    @property
    def num_free(self) -> int:
        return len(self.free)

    def allocate(self) -> int:
        if not self.free:
            raise OutOfBlocks("no free KV cache blocks")
        b = self.free.popleft()
        self.refcount[b] = 1
        return b

    def share(self, b: int) -> int:
        if self.refcount[b] <= 0:
            raise ValueError(f"block {b} is not allocated")
        self.refcount[b] += 1
        return b

    def release(self, b: int) -> None:
        if self.refcount[b] <= 0:
            raise ValueError(f"double free of block {b}")
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            self.free.append(b)


class BlockManager:
    """Grows and frees sequences' block tables."""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)

    @property
    def num_free(self) -> int:
        return self.allocator.num_free

    def blocks_needed(self, seq: Sequence, num_tokens: int) -> int:
        """Extra blocks seq needs to hold its first ``num_tokens`` tokens."""
        need = (num_tokens + self.block_size - 1) // self.block_size
        return max(0, need - len(seq.block_table))

    def can_grow(self, seq: Sequence, num_tokens: int, reserve: int = 0) -> bool:
        return self.blocks_needed(seq, num_tokens) + reserve <= self.allocator.num_free

    def grow(self, seq: Sequence, num_tokens: int) -> None:
        for _ in range(self.blocks_needed(seq, num_tokens)):
            seq.block_table.append(self.allocator.allocate())

    def truncate(self, seq: Sequence, num_tokens: int) -> None:
        """Releases blocks beyond what ``num_tokens`` tokens need."""
        keep = (num_tokens + self.block_size - 1) // self.block_size
        while len(seq.block_table) > keep:
            self.allocator.release(seq.block_table.pop())

    def free(self, seq: Sequence) -> None:
        for b in seq.block_table:
            self.allocator.release(b)
        seq.block_table = []
