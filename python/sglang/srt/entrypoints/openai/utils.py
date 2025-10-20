import logging
from typing import Any, Dict, List, Optional, Union

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    LogProbs,
)

logger = logging.getLogger(__name__)


def to_openai_style_logprobs(
    input_token_logprobs=None,
    output_token_logprobs=None,
    input_top_logprobs=None,
    output_top_logprobs=None,
):
    ret_logprobs = LogProbs()

    # Optimize by batch extending lists instead of per-item append
    def append_token_logprobs(token_logprobs):
        if not token_logprobs:
            return
        # Unzip: get logprob and token_text only, ignore the middle element
        logprobs, _, token_texts = zip(*token_logprobs)
        ret_logprobs.tokens.extend(token_texts)
        ret_logprobs.token_logprobs.extend(logprobs)
        # -1 for each token
        ret_logprobs.text_offset.extend([-1] * len(token_logprobs))

    def append_top_logprobs(top_logprobs):
        if not top_logprobs:
            return
        out = ret_logprobs.top_logprobs
        for tokens in top_logprobs:
            if tokens is not None:
                # More efficient dict-comp, avoid extra comprehension level
                out.append({token[2]: token[0] for token in tokens})
            else:
                out.append(None)

    if input_token_logprobs is not None:
        append_token_logprobs(input_token_logprobs)
    if output_token_logprobs is not None:
        append_token_logprobs(output_token_logprobs)
    if input_top_logprobs is not None:
        append_top_logprobs(input_top_logprobs)
    if output_top_logprobs is not None:
        append_top_logprobs(output_top_logprobs)

    return ret_logprobs


def process_hidden_states_from_ret(
    ret_item: Dict[str, Any],
    request: Union[
        ChatCompletionRequest,
        CompletionRequest,
    ],
) -> Optional[List]:
    """Process hidden states from a ret item in non-streaming response.

    Args:
        ret_item: Response item containing meta_info
        request: The original request object

    Returns:
        Processed hidden states for the last token, or None
    """
    if not request.return_hidden_states:
        return None

    hidden_states = ret_item["meta_info"].get("hidden_states", None)
    if hidden_states is not None:
        hidden_states = hidden_states[-1] if len(hidden_states) > 1 else []
    return hidden_states
