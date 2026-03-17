import torch
from core.kv_cache import BlockAllocator, PhysicalBlock, BlockConfig


class SequenceCache:
    """
    Holds the KV cache for one active sequence.
    block_table[i] = the PhysicalBlock holding logical block i.
    """

    def __init__(self, seq_id: str, allocator: BlockAllocator):
        self.seq_id = seq_id
        self.allocator = allocator
        self.block_table: list[PhysicalBlock] = []
        self.num_cached_tokens: int = 0

    def _tokens_in_last_block(self) -> int:
        block_size = self.allocator.config.block_size
        remainder = self.num_cached_tokens % block_size
        return remainder if remainder != 0 else (block_size if self.block_table else 0)

    def _last_block_full(self) -> bool:
        if not self.block_table:
            return True
        return self._tokens_in_last_block() == self.allocator.config.block_size

    def append_token_kv(
        self,
        layer_idx: int,
        head_idx: int,
        k_vec: torch.Tensor,   # [head_dim]
        v_vec: torch.Tensor,   # [head_dim]
    ) -> None:
        """Write one token's K and V into the cache."""
        if self._last_block_full():
            new_block = self.allocator.allocate()
            self.block_table.append(new_block)

        slot = self._tokens_in_last_block()
        if slot == self.allocator.config.block_size:
            slot = 0

        block = self.block_table[-1]
        block = self.allocator.copy_on_write(block)
        self.block_table[-1] = block

        block.k[layer_idx, head_idx, slot] = k_vec
        block.v[layer_idx, head_idx, slot] = v_vec

    def increment_token_count(self):
        self.num_cached_tokens += 1

    def get_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble K and V for a layer from the block table.
        Returns K, V each of shape [num_heads, num_cached_tokens, head_dim].
        """
        if not self.block_table:
            cfg = self.allocator.config
            empty = torch.zeros(cfg.num_heads, 0, cfg.head_dim, dtype=cfg.dtype)
            return empty, empty

        block_size = self.allocator.config.block_size
        ks, vs = [], []

        for i, block in enumerate(self.block_table):
            if i < len(self.block_table) - 1:
                tokens_in_block = block_size
            else:
                tokens_in_block = self._tokens_in_last_block()
                if tokens_in_block == 0:
                    tokens_in_block = block_size

            ks.append(block.k[layer_idx, :, :tokens_in_block, :])
            vs.append(block.v[layer_idx, :, :tokens_in_block, :])

        K = torch.cat(ks, dim=1)   # [num_heads, total_tokens, head_dim]
        V = torch.cat(vs, dim=1)
        return K, V

    def free(self):
        for block in self.block_table:
            self.allocator.free(block)
        self.block_table.clear()
        self.num_cached_tokens = 0

    def info(self) -> dict:
        return {
            "seq_id": self.seq_id,
            "cached_tokens": self.num_cached_tokens,
            "blocks_used": len(self.block_table),
            "block_ids": [b.block_id for b in self.block_table],
        }


class CacheManager:
    """
    Global manager: creates/destroys SequenceCaches, exposes stats.
    """

    def __init__(self, config: BlockConfig, device: str = "cpu"):
        self.allocator = BlockAllocator(config, device)
        self._sequences: dict[str, SequenceCache] = {}

    def create_sequence(self, seq_id: str) -> SequenceCache:
        if seq_id in self._sequences:
            raise ValueError(f"sequence {seq_id!r} already exists")
        seq = SequenceCache(seq_id, self.allocator)
        self._sequences[seq_id] = seq
        return seq

    def get_sequence(self, seq_id: str) -> SequenceCache:
        return self._sequences[seq_id]

    def free_sequence(self, seq_id: str) -> None:
        seq = self._sequences.pop(seq_id, None)
        if seq:
            seq.free()

    def stats(self) -> dict:
        return {
            "active_sequences": len(self._sequences),
            "sequences": [s.info() for s in self._sequences.values()],
            **self.allocator.stats(),
        }


