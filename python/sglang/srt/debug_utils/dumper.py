import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist


class _Dumper:
    """Utility to dump tensors, which can be useful when comparison checking models.

    Example usage:
    dumper.on_forward_pass_start()
    dumper.dump("layer_start__hidden_states", hidden_states, layer_id=self.layer_id)

    Import from non-SGLang system:
    ```
    import sys
    sys.path.append("/YOUR_PATH/sglang/python/sglang/srt/debug_utils")
    from dumper import dumper
    ```

    Related: `sglang.srt.debug_utils.dump_comparator` for dump comparison
    """

    def __init__(self):
        # Do not import `sglang` to make this file standalone
        self._enable = bool(int(os.environ.get("SGLANG_DUMPER_ENABLE", "1")))
        self._base_dir = Path(os.environ.get("SGLANG_DUMPER_DIR", "/tmp"))
        self._enable_write_file = bool(
            int(os.environ.get("SGLANG_DUMPER_WRITE_FILE", "1"))
        )
        self._partial_name: Optional[str] = None
        self._dump_index = 0
        self._forward_pass_id = 0

    def on_forward_pass_start(self):
        """This should be called on all ranks."""

        if not self._enable:
            return

        # Users may want to `dump` only on some ranks, thus determine name here
        if self._partial_name is None:
            self._partial_name = _get_partial_name()

        self._forward_pass_id += 1
        print(
            f"[Dumper] [{time.time()}] on_forward_pass_start id={self._forward_pass_id}"
        )

    def dump(self, name, value, **kwargs):
        if not self._enable:
            return

        assert (
            self._forward_pass_id >= 1
        ), "Do you forget to call `dumper.on_forward_pass_start()`?"
        assert self._partial_name is not None
        self._dump_index += 1

        rank = _get_rank()
        # Inline full_kwargs construction for lower overhead (avoids new dict/merge)
        full_kwargs_items = (
            ("forward_pass_id", self._forward_pass_id),
            ("rank", rank),
            ("name", name),
            ("dump_index", self._dump_index),
        )
        if kwargs:
            # Merge here only if user provides extra kwargs
            full_kwargs_items += tuple(kwargs.items())
        # Use generator expression for joining, string allocation is single pass
        full_filename = "___".join(f"{k}={v}" for k, v in full_kwargs_items) + ".pt"
        base_dir = self._base_dir
        partial_name = self._partial_name
        path = base_dir / f"sglang_dump_{partial_name}" / full_filename


        sample_value = get_truncated_value(value)


        val_type = type(value)
        # Only check attributes if value is a Tensor
        if isinstance(value, torch.Tensor):
            val_shape = value.shape
            val_dtype = value.dtype
        else:
            val_shape = None
            val_dtype = None

        print(
            f"[Dumper] [{rank}, {time.time()}] {path} "
            f"type={val_type} "
            f"shape={val_shape} "
            f"dtype={val_dtype} "
            f"sample_value={sample_value}"
        )

        if self._enable_write_file:
            # Avoid redundant check/creation of parent directories by locking per call
            parent = path.parent
            if not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
            torch.save(value, str(path))


def _get_partial_name():
    rank = _get_rank()
    object_list = [str(time.time()) if rank == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(object_list, device="cuda")
    return object_list[0]


def _get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_truncated_value(value):
    if value is None:
        return None

    if isinstance(value, tuple):
        return [get_truncated_value(x) for x in value]

    if not isinstance(value, torch.Tensor):
        return None

    numel = value.numel()
    if numel < 200:
        return value

    # Optimize slice creation: avoid constructing tuple for small dimensions
    shape = value.shape
    slices = []
    slice_obj = slice(0, 5)
    for dim_size in shape:
        if dim_size > 200:
            slices.append(slice_obj)
        else:
            slices.append(slice(None))
    return value[tuple(slices)]


dumper = _Dumper()
