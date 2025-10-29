from __future__ import annotations

import abc
import weakref
from typing import TYPE_CHECKING, Optional, Set, Type

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


class BatchedPenalizerOrchestrator:
    def __init__(
        self,
        vocab_size: int,
        batch: ScheduleBatch,
        penalizers: Set[Type["_BatchedPenalizer"]],
    ):
        self.vocab_size = vocab_size
        # Store batch as weakref for memory efficiency
        self._batch_ref = weakref.ref(batch)
        self.device = batch.device

        # Preallocate penalizer list and compute is_required loop inline
        penalizer_objs = []
        is_required = False
        for Penalizer in penalizers:
            penalizer = Penalizer(self)
            penalizer_objs.append((Penalizer, penalizer))
            if penalizer.prepare_if_required():
                is_required = True

        # Construct penalizer mapping only once for performance
        self.penalizers = {pen_type: pen_obj for pen_type, pen_obj in penalizer_objs}
        self.is_required = is_required

    @property
    def batch(self) -> ScheduleBatch | None:
        return self._batch_ref()

    @batch.setter
    def batch(self, value: Optional[ScheduleBatch]):
        if value is None:
            self._batch_ref = lambda: None
        else:
            self._batch_ref = weakref.ref(value)

    def reqs(self):
        # Use weakref to access batch for possible improved GC behavior
        batch = self._batch_ref()
        return batch.reqs if batch is not None else None

    def cumulate_output_tokens(self, output_ids: torch.Tensor):
        """
        Feed the output tokens to the penalizers.

        Args:
            output_ids (torch.Tensor): The output tokens.
        """
        for penalizer in self.penalizers.values():
            penalizer.cumulate_output_tokens(output_ids=output_ids)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the penalizers to the logits.
        Note that it may apply the penalizers in-place.

        Args:
            logits (torch.Tensor): The logits to apply the penalizers to.

        Returns:
            torch.Tensor: The logits after applying the penalizers.
        """
        for penalizer in self.penalizers.values():
            penalizer.apply(logits)

    def filter(self, keep_indices: torch.Tensor):
        """
        Filter the penalizers based on the indices to keep in the batch.

        Args:
            keep_indices (torch.Tensor): Tensor of indices to keep in the batch.
        """
        if not self.is_required:
            return

        if len(keep_indices) == 0:
            self.is_required = False
            for penalizer in self.penalizers.values():
                penalizer.teardown()
            return

        is_required = False
        for penalizer in self.penalizers.values():
            tmp_is_required = penalizer.is_required()
            is_required |= tmp_is_required
            if tmp_is_required:
                penalizer.filter(keep_indices=keep_indices)
            else:
                penalizer.teardown()
        self.is_required = is_required

    def merge(self, their: "BatchedPenalizerOrchestrator"):
        """
        Merge the penalizers of another orchestrator into this one.

        Note that this function **must** be called _before_ self.batch.reqs is updated (filtered).
        Each unprepared penalizers would have to be prepared (creating tensors, etc.) first before merging.
        This step requires the original batch.reqs, before it gets merged with other batch.reqs.

        Args:
            their (BatchedPenalizerOrchestrator): The orchestrator to merge into this one.
        """
        if not self.is_required and not their.is_required:
            return

        self.is_required = True
        for penalizer, their_penalizer in their.penalizers.items():
            self.penalizers[penalizer].merge(their_penalizer)


class _BatchedPenalizer(abc.ABC):
    """
    An abstract class for a batched penalizer.
    """

    def is_prepared(self) -> bool:
        return self._is_prepared

    def is_required(self) -> bool:
        return self._is_required()

    def prepare(self):
        if not self._is_prepared:
            self._prepare()
            self._is_prepared = True

    def prepare_if_required(self):
        if self._is_required():
            self.prepare()
            return True
        else:
            return False

    def teardown(self):
        self._is_prepared = False

    def cumulate_output_tokens(self, output_ids: torch.Tensor):
        if not self._is_prepared:
            return

        self._cumulate_output_tokens(output_ids=output_ids)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._is_prepared:
            return

        self._apply(logits=logits)

    def filter(self, keep_indices: torch.Tensor):
        if not self._is_prepared:
            return

        self._filter(keep_indices=keep_indices)

    def merge(self, their: "_BatchedPenalizer"):
        if not self._is_prepared and not their._is_prepared:
            return

        self.prepare()
        their.prepare()
        self._merge(their)

    @abc.abstractmethod
    def _is_required(self) -> bool:
        """
        Check if the penalizer is required to be prepared.
        """
        pass

    @abc.abstractmethod
    def _prepare(self):
        """
        Prepare the penalizer.
        Usually, this is where the penalizer initializes its tensors.
        """
        pass

    @abc.abstractmethod
    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        """
        Cumulate the output tokens.
        Orchestrator will call this function to feed the output tokens to the penalizer.
        """
        pass

    @abc.abstractmethod
    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the penalizer to the logits.
        Penalizers can modify the logits in-place if needed.
        """
        pass

    @abc.abstractmethod
    def _filter(self, keep_indices: torch.Tensor):
        """
        Filter the penalizer (tensors or underlying data) based on the indices to keep in the batch.
        """
        pass

    @abc.abstractmethod
    def _merge(self, their: "_BatchedPenalizer"):
        """
        Merge the penalizer with another penalizer.
        """
        pass
