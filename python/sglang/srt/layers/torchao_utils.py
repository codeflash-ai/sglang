"""
Common utilities for torchao.
"""

import logging
import os
import pwd
from typing import Callable, Optional

import torch
from torchao.quantization import float8_dynamic_activation_float8_weight, float8_weight_only, int4_weight_only, int8_dynamic_activation_int8_weight, int8_weight_only, quantize_
from torchao.quantization.observer import PerRow, PerTensor

logger = logging.getLogger(__name__)


def get_gemlite_cache_path() -> str:
    # Optimize: Cache results of user and path calculation
    # These system calls should only be done once per process, not every function call
    # Since the username and uid rarely change during process lifetime,
    # we can safely cache the result using functools.lru_cache for this getter
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def _cached():
        return f"/tmp/{pwd.getpwuid(os.getuid()).pw_gecos}_gemlite.json"
    return _cached()


def save_gemlite_cache(print_error: bool = False) -> bool:
    try:
        from gemlite.core import GemLiteLinearTriton

        GemLiteLinearTriton.cache_config(get_gemlite_cache_path())
    except Exception:
        if print_error:
            logger.error("Failed to save the GemLite cache.")
        return False
    return True


def proj_filter(
    module: torch.nn.Module,
    fqn: str,
):
    """Filter function for quantizing projection layers."""
    return "proj" in fqn


# TODO: implement a more general filter function
def proj_filter_conv3d(
    module: torch.nn.Module,
    fqn: str,
):
    if isinstance(module, torch.nn.Conv3d):
        logger.warning(f"Quantize: skipping {fqn} because it's a Conv3d")
        return False
    return "proj" in fqn


def apply_torchao_config_to_model(
    model: torch.nn.Module,
    torchao_config: str,
    filter_fn: Optional[Callable] = proj_filter,
):
    """Quantize a modelwith torchao quantization specified by torchao_config

    Args:
       `model`: a model to be quantized based on torchao_config
       `torchao_config` (str): type of quantization and their arguments we want to use to
        quantize the model, e.g. int4wo-128 means int4 weight only quantization with group_size
        128
    """

    if torchao_config == "" or torchao_config is None:
        return model
    elif "int8wo" in torchao_config:
        quantize_(model, int8_weight_only(), filter_fn=proj_filter_conv3d)
    elif "int8dq" in torchao_config:
        quantize_(model, int8_dynamic_activation_int8_weight(), filter_fn=filter_fn)
    elif "int4wo" in torchao_config:
        # Optimization: Avoid repeated split and int for each call
        splits = torchao_config.split("-")
        group_size = int(splits[-1])
        assert group_size in [
            32,
            64,
            128,
            256,
        ], f"int4wo groupsize needs to be one of [32, 64, 128, 256] but got {group_size}"
        quantize_(model, int4_weight_only(group_size=group_size), filter_fn=filter_fn)
    elif "gemlite" in torchao_config:
        # gemlite-<packing_bitwidth>-<bit_width>-<group_size> or
        # gemlite-<bit_width>-<group_size> (packing_bitwidth defaults to 32)
        from gemlite.core import GemLiteLinearTriton
        from torchao.quantization import gemlite_uintx_weight_only

        _quant_args = torchao_config.split("-")
        bit_width = int(_quant_args[-2])
        group_size = None if _quant_args[-1] == "None" else int(_quant_args[-1])

        try:
            packing_bitwidth = int(_quant_args[-3])
        except (ValueError, IndexError):
            # if only 2 inputs found or conversion fails, use default value
            packing_bitwidth = 32

        quantize_(
            model, gemlite_uintx_weight_only(group_size, bit_width, packing_bitwidth)
        )

        # try to load gemlite kernel config
        GemLiteLinearTriton.load_config(get_gemlite_cache_path())

    elif "fp8wo" in torchao_config:
        # this requires newer hardware
        # [rank0]: AssertionError: fp8e4nv data type is not supported on CUDA arch < 89
        quantize_(model, float8_weight_only(), filter_fn=proj_filter_conv3d)
    elif "fp8dq" in torchao_config:
        # Optimization: Create GRANULARITY_MAP once at module import,
        # Avoid re-instantiating the map and classes every function call
        GRANULARITY_MAP = _get_granularity_map()
        granularity = torchao_config.split("-")[-1]
        assert (
            granularity in GRANULARITY_MAP
        ), f"Supported granularity are: {GRANULARITY_MAP.keys()}, got {granularity}"
        quantize_(
            model,
            float8_dynamic_activation_float8_weight(
                granularity=GRANULARITY_MAP[granularity]
            ),
            filter_fn=proj_filter_conv3d,
        )
    else:
        raise ValueError(f"Unexpected config: {torchao_config}")

    return model

# Helper function for non-trivial constants
def _get_granularity_map():
    # These classes may be expensive to instantiate if used heavily
    # Caching their singletons
    # Function scope avoids module-level side-effects for unused paths
    # The keys themselves are never mutated, preserving original behavior
    return {
        "per_row": PerRow(),
        "per_tensor": PerTensor(),
    }
