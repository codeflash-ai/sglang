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

    # Combine logprob sources if present, to minimize function call overhead and reduce attribute access
    token_logprobs_sources = []
    if input_token_logprobs is not None:
        token_logprobs_sources.append(input_token_logprobs)
    if output_token_logprobs is not None:
        token_logprobs_sources.append(output_token_logprobs)

    # Use batch appends for efficiency
    tokens = ret_logprobs.tokens
    token_logprobs = ret_logprobs.token_logprobs
    text_offset = ret_logprobs.text_offset

    for logprobs in token_logprobs_sources:
        # Use local variables to avoid attribute lookups in loop
        append = tokens.append
        append_logprob = token_logprobs.append
        append_offset = text_offset.append
        for logprob, _, token_text in logprobs:
            append(token_text)
            append_logprob(logprob)
            # Not supported yet
            append_offset(-1)

    # Similarly process top logprobs in two stages
    top_logprobs_sources = []
    if input_top_logprobs is not None:
        top_logprobs_sources.append(input_top_logprobs)
    if output_top_logprobs is not None:
        top_logprobs_sources.append(output_top_logprobs)

    top_logprobs_attr = ret_logprobs.top_logprobs
    for top_logprobs in top_logprobs_sources:
        # Fast-path append
        append_top = top_logprobs_attr.append
        for tokens in top_logprobs:
            if tokens is not None:
                # Compose dict directly without comprehension for slightly better perf
                d = {}
                for token in tokens:
                    d[token[2]] = token[0]
                append_top(d)
            else:
                append_top(None)

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
