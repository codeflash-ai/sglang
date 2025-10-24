import contextvars
import inspect
import logging
import os
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch

from sglang.srt.compilation.compilation_config import CompilationConfig

logger = logging.getLogger(__name__)

_COMPILE_ENABLED = contextvars.ContextVar("_COMPILE_ENABLED", default=False)


@contextmanager
def set_compiled(enabled: bool = True):
    token = _COMPILE_ENABLED.set(enabled)
    try:
        yield
    finally:
        _COMPILE_ENABLED.reset(token)


@dataclass
class IntermediateTensors:
    """For all pipeline stages except the last, we need to return the hidden
    states and residuals to be sent to the next stage. This data structure
    contains the hidden states and residuals for a request.

    Each stage also needs to handle its own finished_sending and
    finished_recving in case of kv transfer.
    """

    tensors: dict[str, torch.Tensor]
    # [req_ids]
    finished_sending: Optional[set[str]] = None
    finished_recving: Optional[set[str]] = None

    def __init__(self, tensors):
        # manually define this function, so that
        # Dynamo knows `IntermediateTensors()` comes from this file.
        # Otherwise, dataclass will generate this function by evaluating
        # a string, and we will lose the information about the source file.
        self.tensors = tensors

    def __getitem__(self, key: Union[str, slice]):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})

    def __setitem__(self, key: str, value: torch.Tensor):
        self.tensors[key] = value

    def items(self):
        return self.tensors.items()

    def __len__(self):
        return len(self.tensors)

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self

    def __repr__(self) -> str:
        return f"IntermediateTensors(tensors={self.tensors})"


def _normalize_dims(dims, ndim: int):
    dims = [dims] if isinstance(dims, int) else list(dims)
    return [d if d >= 0 else ndim + d for d in dims]


class _MaybeIntermediateTensors:
    """Duck-typed check to support your IntermediateTensors without importing."""

    def __init__(self, obj):
        self.is_intermediate = hasattr(obj, "tensors") and isinstance(
            getattr(obj, "tensors"), dict
        )
        self.obj = obj


def _mark_dynamic_on_value(val, dims):
    if isinstance(val, torch.Tensor):
        torch._dynamo.mark_dynamic(val, _normalize_dims(dims, val.ndim))
    else:
        mit = _MaybeIntermediateTensors(val)
        if mit.is_intermediate:
            for t in mit.obj.tensors.values():
                torch._dynamo.mark_dynamic(t, _normalize_dims(dims, t.ndim))
        # else: ignore (None or non-tensor)


def _infer_dynamic_arg_dims_from_annotations(forward_fn):
    sig = inspect.signature(forward_fn)
    parameters = sig.parameters.items()
    dyn = {}

    # Precompute commonly used names to avoid attribute lookups
    INTERMEDIATE_NAME = "IntermediateTensors"
    TENSOR_NAME = "Tensor"
    TENSOR_TYPE = torch.Tensor

    for name, p in parameters:
        ann = p.annotation

        # Short circuit exact torch.Tensor type
        if ann is TENSOR_TYPE:
            dyn[name] = 0
            continue

        # Avoid repeated getattr on __args__ by local variable
        ann_args = getattr(ann, "__args__", None)

        # Fast path for Optional[torch.Tensor] and similar (e.g., typing.Optional[torch.Tensor], Union[torch.Tensor, NoneType])
        if ann_args:
            first_arg = ann_args[0]
            if getattr(first_arg, "__name__", "") == TENSOR_NAME:
                dyn[name] = 0
                continue

        # Test IntermediateTensors direct annotation
        ann_name = getattr(ann, "__name__", "")
        if ann_name == INTERMEDIATE_NAME:
            dyn[name] = 0
            continue

        # Only check args if ann_args is not None/empty
        if ann_args:
            # Avoid unneeded function call to any() in most cases with a for-loop and break
            for a in ann_args:
                if getattr(a, "__name__", "") == INTERMEDIATE_NAME:
                    dyn[name] = 0
                    break

    if not dyn:
        raise ValueError("No dynamic dims inferred; pass dynamic_arg_dims explicitly.")
    return dyn


def install_torch_compiled(
    module: torch.nn.Module,
    *,
    dynamic_arg_dims: dict[str, Union[int, list[int]]] | None = None,
    backend_factory: Optional[Callable[[torch.fx.GraphModule, list], Callable]] = None,
    compile_config: CompilationConfig = None,
    fullgraph: bool = True,
    graph_pool: Any = None,
):
    unbound_fwd = module.__class__.forward
    if not callable(unbound_fwd):
        raise TypeError("module.__class__.forward must be callable")
    original_code = unbound_fwd.__code__

    dyn_map = dynamic_arg_dims or _infer_dynamic_arg_dims_from_annotations(unbound_fwd)

    if backend_factory is None:
        from sglang.srt.compilation.backend import SGLangBackend

        backend_factory = lambda gm, ex: SGLangBackend(compile_config, graph_pool)(
            gm, ex
        )

    compiled_codes: list[type(original_code)] = []
    state = {"compiled": False, "compiled_callable": None}

    def bytecode_hook(old_code, new_code):
        if old_code is not original_code:
            return
        frame = sys._getframe()
        while frame and frame.f_back:
            frame = frame.f_back
            if (
                frame.f_code.co_name == "_compile"
                and os.path.basename(frame.f_code.co_filename) == "convert_frame.py"
            ):
                break
        try:
            dynamo_frame = frame.f_locals["frame"]
        except Exception:
            return
        if dynamo_frame.f_code is not old_code:
            return
        if dynamo_frame.f_locals.get("self") is not module:
            return
        compiled_codes.append(new_code)

    torch._dynamo.convert_frame.register_bytecode_hook(bytecode_hook)

    def _ensure_compiled(self, *args, **kwargs):
        """Compile on first use (with flag ON)."""
        if state["compiled"]:
            return
        # Mark dynamic dims only when we are about to compile
        sig = inspect.signature(unbound_fwd)
        ba = sig.bind(self, *args, **kwargs)
        ba.apply_defaults()
        for name, dims in (dyn_map or {}).items():
            if name in ba.arguments:
                val = ba.arguments[name]
                if val is not None:
                    _mark_dynamic_on_value(val, dims)

        # Avoid cross-instance cache reuse
        torch._dynamo.eval_frame.remove_from_cache(unbound_fwd.__code__)

        bound = types.MethodType(unbound_fwd, self)
        compiled_callable = torch.compile(
            bound, fullgraph=fullgraph, backend=backend_factory
        )

        # Trigger Dynamo so bytecode hook can capture
        compiled_callable(*args, **kwargs)

        state["compiled"] = True
        state["compiled_callable"] = compiled_callable

    def trampoline(self, *args, **kwargs):
        use_compiled = _COMPILE_ENABLED.get()
        if use_compiled:
            if not state["compiled"]:
                _ensure_compiled(self, *args, **kwargs)

            compiled_callable = state["compiled_callable"]
            return compiled_callable(*args, **kwargs)
        else:
            # Explicitly run the original uncompiled forward
            return unbound_fwd(self, *args, **kwargs)

    module.forward = types.MethodType(trampoline, module)
    return module
