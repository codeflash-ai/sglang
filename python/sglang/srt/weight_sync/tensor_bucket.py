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
    # Assuming definition is provided elsewhere in the codebase
    # Placeholder for type checking
    def __init__(
        self,
        name: str,
        shape,
        dtype,
        start_idx: int,
        end_idx: int,
        numel: int,
    ):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.numel = numel


class FlattenedTensorBucket:
    """
    A bucket that flattens multiple tensors into a single tensor for efficient processing
    while preserving all metadata needed for reconstruction.
    """

    def __init__(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]] = None,
        flattened_tensor: torch.Tensor = None,
        metadata: List['FlattenedTensorMetadata'] = None,
    ):
        """
        Initialize a tensor bucket from a list of named tensors OR from pre-flattened data.
        Args:
            named_tensors: List of (name, tensor) tuples (for creating new bucket)
            flattened_tensor: Pre-flattened tensor (for reconstruction)
            metadata: Pre-computed metadata (for reconstruction)
        """
        if named_tensors is not None:
            # Create bucket from named tensors
            if not named_tensors:
                raise ValueError("Cannot create empty tensor bucket")

            num_tensors = len(named_tensors)
            self.metadata: List[FlattenedTensorMetadata] = [None] * num_tensors
            flattened_tensors = []
            current_idx = 0

            # Precompute total numel and use it to streamline cat/op allocation
            total_numel = 0
            for _, tensor in named_tensors:
                total_numel += tensor.numel()

            # Preallocate output Tensor on the appropriate device/dtype if everything is consistent
            # This block attempts to avoid multiple small allocations and use slicing
            # Only if all tensors are of the same dtype and device
            # Otherwise, default to original logic
            all_same_dtype = True
            all_same_device = True
            ref_dtype = named_tensors[0][1].dtype
            ref_device = named_tensors[0][1].device
            for _, tensor in named_tensors:
                if tensor.dtype != ref_dtype:
                    all_same_dtype = False
                if tensor.device != ref_device:
                    all_same_device = False

            fast_path = all_same_dtype and all_same_device

            if fast_path and num_tensors > 1:
                out = torch.empty(total_numel, dtype=ref_dtype, device=ref_device)
                for i, (name, tensor) in enumerate(named_tensors):
                    numel = tensor.numel()
                    flat_tensor = tensor.flatten()
                    # Write in-place
                    out[current_idx : current_idx + numel] = flat_tensor
                    self.metadata[i] = FlattenedTensorMetadata(
                        name=name,
                        shape=tensor.shape,
                        dtype=tensor.dtype,
                        start_idx=current_idx,
                        end_idx=current_idx + numel,
                        numel=numel,
                    )
                    current_idx += numel
                self.flattened_tensor = out
            else:
                flattened_tensors = []
                for i, (name, tensor) in enumerate(named_tensors):
                    flat = tensor.flatten()
                    flattened_tensors.append(flat)
                    numel = flat.numel()
                    self.metadata[i] = FlattenedTensorMetadata(
                        name=name,
                        shape=tensor.shape,
                        dtype=tensor.dtype,
                        start_idx=current_idx,
                        end_idx=current_idx + numel,
                        numel=numel,
                    )
                    current_idx += numel
                if num_tensors == 1:
                    self.flattened_tensor = flattened_tensors[0]
                else:
                    self.flattened_tensor = torch.cat(flattened_tensors, dim=0)
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
