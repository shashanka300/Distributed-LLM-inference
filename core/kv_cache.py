import torch
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class BlockConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    block_size: int = 16
    num_blocks: int = 512
    dtype: torch.dtype = torch.float16


class PhysicalBlock:
    """One fixed-size slab of memory holding KV vectors for block_size tokens."""

    def __init__(self, block_id: int, config: BlockConfig, device: str):
        self.block_id = block_id
        self.ref_count = 0
        self.prefix_hash: Optional[int] = None

        shape = (config.num_layers, config.num_heads, config.block_size, config.head_dim)
        self.k = torch.zeros(shape, dtype=config.dtype, device=device)
        self.v = torch.zeros(shape, dtype=config.dtype, device=device)


class BlockAllocator:
    """
    Manages a fixed pool of PhysicalBlocks.
    Free list gives O(1) alloc and free.
    Thread-safe via a single lock.
    """

    def __init__(self, config: BlockConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self._lock = threading.Lock()

        # pre-allocate entire pool upfront
        self._pool = [
            PhysicalBlock(i, config, device)
            for i in range(config.num_blocks)
        ]

        # Free list used as a stack (LIFO for better cache locality).
        self._free: list[int] = list(range(config.num_blocks))

        # prefix cache: hash -> block_id
        self._prefix_cache: dict[int, int] = {}

    # Allocation

    def allocate(self) -> PhysicalBlock:
        with self._lock:
            if not self._free:
                raise MemoryError(
                    f"KV cache exhausted: all {self.config.num_blocks} blocks in use."
                )
            block_id = self._free.pop()
            block = self._pool[block_id]
            block.ref_count = 1
            block.prefix_hash = None
            return block

    def free(self, block: PhysicalBlock) -> None:
        with self._lock:
            block.ref_count -= 1
            if block.ref_count == 0:
                if block.prefix_hash is not None:
                    self._prefix_cache.pop(block.prefix_hash, None)
                    block.prefix_hash = None
                self._free.append(block.block_id)

    # Prefix caching

    def get_or_create_prefix_block(
        self, prefix_hash: int
    ) -> tuple["PhysicalBlock", bool]:
        """
        Returns (block, cache_hit).
        On a hit: increments ref_count and returns the existing block.
        On a miss: allocates a fresh block, registers hash, returns it.
        """
        with self._lock:
            if prefix_hash in self._prefix_cache:
                block_id = self._prefix_cache[prefix_hash]
                block = self._pool[block_id]
                block.ref_count += 1
                return block, True

            if not self._free:
                raise MemoryError("KV cache exhausted")
            block_id = self._free.pop()
            block = self._pool[block_id]
            block.ref_count = 1
            block.prefix_hash = prefix_hash
            self._prefix_cache[prefix_hash] = block_id
            return block, False

    # Copy-on-write

    def copy_on_write(self, block: PhysicalBlock) -> PhysicalBlock:
        """
        If block is shared (ref_count > 1), allocate a fresh block and copy.
        This mirrors what the OS does when a forked process writes to a
        shared page. Returns the block the caller should now write to.
        """
        if block.ref_count <= 1:
            return block  # not shared, safe to write in place

        # Allocate without calling allocate(); we already hold lock here.
        with self._lock:
            if not self._free:
                raise MemoryError("KV cache exhausted during copy-on-write")
            new_id = self._free.pop()
            new_block = self._pool[new_id]
            new_block.ref_count = 1
            new_block.prefix_hash = None

        new_block.k.copy_(block.k)
        new_block.v.copy_(block.v)

        # decrement old block ref
        with self._lock:
            block.ref_count -= 1
            if block.ref_count == 0:
                if block.prefix_hash is not None:
                    self._prefix_cache.pop(block.prefix_hash, None)
                    block.prefix_hash = None
                self._free.append(block.block_id)

        return new_block

    # Diagnostics

    def stats(self) -> dict:
        with self._lock:
            used = self.config.num_blocks - len(self._free)
            return {
                "total_blocks": self.config.num_blocks,
                "used_blocks": used,
                "free_blocks": len(self._free),
                "utilization": round(used / self.config.num_blocks, 3),
                "prefix_cache_entries": len(self._prefix_cache),
            }


