from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class FlattenedTensorMetadata:
    """Metadata for a tensor in a flattened bucket"""

    name: str
    shape: torch.Size
    dtype: torch.dtype
    start_idx: int
    end_idx: int
    numel: int


class FlattenedTensorBucket:
    """
    A bucket that flattens multiple tensors into a single tensor for efficient processing
    while preserving all metadata needed for reconstruction.
    """

    def __init__(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]] = None,
        flattened_tensor: torch.Tensor = None,
        metadata: List['FlattenedTensorMetadata'] = None,  # quote type for annotation compatibility
    ):
        """
        Initialize a tensor bucket from a list of named tensors OR from pre-flattened data.
        Args:
            named_tensors: List of (name, tensor) tuples (for creating new bucket)
            flattened_tensor: Pre-flattened tensor (for reconstruction)
            metadata: Pre-computed metadata (for reconstruction)
        """
        if named_tensors is not None:
            if not named_tensors:
                raise ValueError("Cannot create empty tensor bucket")

            # Preallocate metadata and flattened tensor list
            num_tensors = len(named_tensors)
            self.metadata: List['FlattenedTensorMetadata'] = [None] * num_tensors

            # Use a generator to avoid an intermediate list and unnecessary flatten
            flattened_tensors_iter = (
                tensor.flatten() for _, tensor in named_tensors
            )
            # Flatten all first and stash for efficient index computation
            flattened_tensors = []
            current_idx = 0
            for i, ((name, tensor), ft) in enumerate(zip(named_tensors, flattened_tensors_iter)):
                numel = ft.numel()
                self.metadata[i] = FlattenedTensorMetadata(
                    name=name,
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    start_idx=current_idx,
                    end_idx=current_idx + numel,
                    numel=numel,
                )
                flattened_tensors.append(ft)
                current_idx += numel

            # Concatenate all flattened tensors
            self.flattened_tensor: torch.Tensor = torch.cat(flattened_tensors, dim=0)
        else:
            # Initialize from pre-flattened data
            if flattened_tensor is None or metadata is None:
                raise ValueError(
                    "Must provide either named_tensors or both flattened_tensor and metadata"
                )
            self.flattened_tensor = flattened_tensor
            self.metadata = metadata

    def get_flattened_tensor(self) -> torch.Tensor:
        """Get the flattened tensor containing all bucket tensors"""
        return self.flattened_tensor

    def get_metadata(self) -> List[FlattenedTensorMetadata]:
        """Get metadata for all tensors in the bucket"""
        return self.metadata

    def reconstruct_tensors(self) -> List[Tuple[str, torch.Tensor]]:
        """
        Reconstruct original tensors from flattened tensor with optimized performance.
        Uses memory-efficient operations to minimize allocations and copies.
        """
        # preallocate the result list
        reconstructed = [None] * len(self.metadata)

        for i, meta in enumerate(self.metadata):
            tensor = self.flattened_tensor[meta.start_idx : meta.end_idx].reshape(
                meta.shape
            )

            # batch dtype conversion (if needed)
            if tensor.dtype != meta.dtype:
                tensor = tensor.to(meta.dtype)

            reconstructed[i] = (meta.name, tensor)

        return reconstructed
