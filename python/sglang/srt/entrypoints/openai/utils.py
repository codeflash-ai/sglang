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

    def append_token_logprobs(token_logprobs):
        if token_logprobs:
            tokens = [tpl[2] for tpl in token_logprobs]
            logprobs = [tpl[0] for tpl in token_logprobs]
            # -1 is used for all offsets
            offsets = [-1] * len(token_logprobs)
            ret_logprobs.tokens.extend(tokens)
            ret_logprobs.token_logprobs.extend(logprobs)
            ret_logprobs.text_offset.extend(offsets)

    def append_top_logprobs(top_logprobs):
        if top_logprobs:
            # List comp for speed, batch-appending results
            extended = [
                {token[2]: token[0] for token in tokens} if tokens is not None else None
                for tokens in top_logprobs
            ]
            ret_logprobs.top_logprobs.extend(extended)

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
