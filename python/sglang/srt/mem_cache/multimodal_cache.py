import logging
from collections import OrderedDict

import torch

# Set up logging for cache behavior
logger = logging.getLogger(__name__)


class MultiModalCache:
    """MultiModalCache is used to store vlm encoder results with LRU eviction"""

    def __init__(
        self,
        max_size: int,
    ):
        self.max_size = max_size
        self.mm_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.current_size = 0

    def _allocate(self, embedding_size: int) -> bool:
        """Allocate space by evicting least recently used entries"""
        evictions = 0
        cache = self.mm_cache
        current_size = self.current_size
        max_size = self.max_size

        # Precompute the required free space
        required_free = current_size + embedding_size - max_size
        # Early exit if no evictions required
        if required_free <= 0:
            return True

        # Collect embeddings until we free enough space (minimize repeated computation)
        pop = cache.popitem
        get_tensor_size = torch.Tensor.numel  # Avoid repeated attribute lookup
        element_size = None

        freed = 0
        while current_size + embedding_size > max_size and cache:
            _, old_embedding = pop(last=False)
            # Cache element_size value for all tensors - usually all tensors have the same dtype so this is an optimization
            if element_size is None:
                element_size = old_embedding.element_size()
            evicted_size = get_tensor_size(old_embedding) * element_size
            current_size -= evicted_size
            freed += evicted_size

        self.current_size = current_size
        evictions = freed

        if evictions > 0:
            logger.debug(
                f"Cache eviction: evicted {evictions} bytes, remaining size: {self.current_size}/{self.max_size} bytes"
            )

        if self.current_size + embedding_size > self.max_size:
            return False
        return True

    def put(self, mm_hash: int, embedding: torch.Tensor) -> bool:
        data_size = self._get_tensor_size(embedding)
        # Lazy free cache if not enough space
        if not self._allocate(data_size):
            return False
        self.mm_cache[mm_hash] = embedding
        self.current_size += data_size
        return True

    def has(self, mm_hash: int) -> bool:
        return mm_hash in self.mm_cache

    def get(self, mm_hash: int) -> torch.Tensor:
        """Get embedding and update LRU order"""
        if mm_hash in self.mm_cache:
            # Move to end (most recently used)
            self.mm_cache.move_to_end(mm_hash)
            return self.mm_cache[mm_hash]
        return None

    def clear(self):
        self.mm_cache.clear()
        self.current_size = 0

    def _get_tensor_size(self, embedding: torch.Tensor):
        return embedding.element_size() * embedding.numel()

    def __len__(self):
        return len(self.mm_cache)
