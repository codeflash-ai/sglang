from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, final

from sglang.srt.entrypoints.openai.protocol import UsageInfo


@final
class UsageProcessor:
    """Stateless helpers that turn raw token counts into a UsageInfo."""

    @staticmethod
    def _details_if_cached(count: int) -> Optional[Dict[str, int]]:
        """Return {"cached_tokens": N} only when N > 0 (keeps JSON slim)."""
        return {"cached_tokens": count} if count > 0 else None

    @staticmethod
    def calculate_response_usage(
        responses: List[Dict[str, Any]],
        n_choices: int = 1,
        enable_cache_report: bool = False,
    ) -> UsageInfo:
        # Local variables for fast attribute access
        responses_len = len(responses)
        meta_infos = [r["meta_info"] for r in responses]

        # Compute completion_tokens with efficient field extraction
        completion_tokens = sum(m["completion_tokens"] for m in meta_infos)

        # Compute prompt_tokens efficiently without repeated indexing
        if n_choices == 1:
            # Fast path: one loop, no step math
            prompt_tokens = sum(m["prompt_tokens"] for m in meta_infos)
        else:
            # Avoid repeated lookup in hot inner loop
            prompt_tokens = 0
            for i in range(0, responses_len, n_choices):
                prompt_tokens += meta_infos[i]["prompt_tokens"]


        cached_details = None
        if enable_cache_report:
            cached_total = sum(m.get("cached_tokens", 0) for m in meta_infos)
            cached_details = UsageProcessor._details_if_cached(cached_total)

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_details,
        )

    @staticmethod
    def calculate_streaming_usage(
        prompt_tokens: Mapping[int, int],
        completion_tokens: Mapping[int, int],
        cached_tokens: Mapping[int, int],
        n_choices: int,
        enable_cache_report: bool = False,
    ) -> UsageInfo:
        # index % n_choices == 0 marks the first choice of a prompt
        total_prompt_tokens = sum(
            tok for idx, tok in prompt_tokens.items() if idx % n_choices == 0
        )
        total_completion_tokens = sum(completion_tokens.values())

        cached_details = (
            UsageProcessor._details_if_cached(sum(cached_tokens.values()))
            if enable_cache_report
            else None
        )

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            cached_tokens=cached_details,
        )

    @staticmethod
    def calculate_token_usage(
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: Optional[Dict[str, int]] = None,
    ) -> UsageInfo:
        """Calculate token usage information"""
        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=cached_tokens,
        )
