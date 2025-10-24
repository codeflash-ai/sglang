import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import orjson
from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import (
    _find_common_prefix,
    _is_complete_json,
    _partial_json_loads,
)

logger = logging.getLogger(__name__)


class BaseFormatDetector(ABC):
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(self):
        # Streaming state management
        self._buffer = ""
        self.prev_tool_call_arr: List[Dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: List[str] = []

        self.bot_token = ""
        self.eot_token = ""
        self.tool_call_separator = ", "

        # Build tool indices and cache for instance
        self._tool_indices: Dict[str, int] = {}
        self._tools_ref: List[Tool] = []
        # Cache JSON serializations for each tool call to minimize repeated json.dumps
        self._prev_args_jsons: List[str] = []

    def _get_tool_indices(self, tools: List[Tool]) -> Dict[str, int]:
        """
        Get a mapping of tool names to their indices in the tools list.

        This utility method creates a dictionary mapping function names to their
        indices in the tools list, which is commonly needed for tool validation
        and ToolCallItem creation.

        Args:
            tools: List of available tools

        Returns:
            Dictionary mapping tool names to their indices
        """
        return {
            tool.function.name: i for i, tool in enumerate(tools) if tool.function.name
        }

    def parse_base_json(self, action: Any, tools: List[Tool]) -> List[ToolCallItem]:
        tool_indices = self._get_tool_indices(tools)
        if not isinstance(action, list):
            action = [action]

        results = []
        for act in action:
            name = act.get("name")
            if name and name in tool_indices:
                results.append(
                    ToolCallItem(
                        tool_index=-1,  # Caller should update this based on the actual tools array called
                        name=name,
                        parameters=json.dumps(
                            act.get("parameters") or act.get("arguments", {}),
                            ensure_ascii=False,
                        ),
                    )
                )
            else:
                logger.warning(f"Model attempted to call undefined function: {name}")

        return results

    @abstractmethod
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parses the text in one go. Returns success=True if the format matches, otherwise False.
        Note that leftover_text here represents "content that this parser will not consume further".
        """
        action = orjson.loads(text)
        return StreamingParseResult(calls=self.parse_base_json(action, tools))

    def _ends_with_partial_token(self, buffer: str, bot_token: str) -> int:
        """
        Check if buffer ends with a partial bot_token.
        Return the length of the partial bot_token.

        For some format, the bot_token is not a token in model's vocabulary, such as
        `[TOOL_CALLS] [` in Mistral.
        """
        for i in range(1, min(len(buffer) + 1, len(bot_token))):
            if bot_token.startswith(buffer[-i:]):
                return i
        return 0

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
               Streaming incremental parsing with tool validation.

               This base implementation works best with formats where:
               1. bot_token is followed immediately by JSON (e.g., bot_token + JSON_array)
               2. JSON can be parsed incrementally using partial_json_loads
               3. Multiple tool calls are separated by "; " or ", "

               Examples of incompatible formats (need custom implementation, may reuse some logic from this class):
               - Each tool call is wrapped in a separate block: See Qwen25Detector
               - Multiple separate blocks: [TOOL_CALLS] [...]
        [TOOL_CALLS] [...]
               - Tool call is Pythonic style

               For incompatible formats, detectors should override this method with custom logic.
        """
        self._buffer += new_text
        current_text = self._buffer

        if not (
            self.has_tool_call(current_text)
            or (
                self.current_tool_id > 0
                and current_text.startswith(self.tool_call_separator)
            )
        ):
            if not self._ends_with_partial_token(self._buffer, self.bot_token):
                normal_text = self._buffer
                self._buffer = ""
                if self.eot_token in normal_text:
                    normal_text = normal_text.replace(self.eot_token, "")
                return StreamingParseResult(normal_text=normal_text)
            else:
                return StreamingParseResult()

        # Build/rebuild tool indices only if needed
        self._ensure_tool_indices(tools)

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

        try:
            try:
                tool_call_pos = current_text.find(self.bot_token)
                if tool_call_pos != -1:
                    start_idx = tool_call_pos + len(self.bot_token)
                elif self.current_tool_id > 0 and current_text.startswith(
                    self.tool_call_separator
                ):
                    start_idx = len(self.tool_call_separator)
                else:
                    start_idx = 0

                if start_idx >= len(current_text):
                    return StreamingParseResult()

                (obj, end_idx) = _partial_json_loads(current_text[start_idx:], flags)

                is_current_complete = _is_complete_json(
                    current_text[start_idx : start_idx + end_idx]
                )

                if "name" in obj and obj["name"] not in self._tool_indices:
                    self._buffer = ""
                    self.current_tool_id = -1
                    self.current_tool_name_sent = False
                    if self.streamed_args_for_tool:
                        self.streamed_args_for_tool.pop()
                    return StreamingParseResult()

                if "parameters" in obj:
                    assert (
                        "arguments" not in obj
                    ), "model generated both parameters and arguments"
                    obj["arguments"] = obj["parameters"]

                current_tool_call = obj

            except MalformedJSON:
                return StreamingParseResult()

            if not current_tool_call:
                return StreamingParseResult()

            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name and function_name in self._tool_indices:
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.streamed_args_for_tool.append("")
                        self._prev_args_jsons.append("")
                    elif self.current_tool_id >= len(self.streamed_args_for_tool):
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")
                            self._prev_args_jsons.append("")
                    res = StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=function_name,
                                parameters="",
                            )
                        ],
                    )
                    self.current_tool_name_sent = True
                else:
                    res = StreamingParseResult()

            else:
                cur_arguments = current_tool_call.get("arguments")
                res = StreamingParseResult()

                if cur_arguments:
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    # Use cached JSON for previous state; serialize only if changed
                    cur_args_json = json.dumps(cur_arguments, separators=(",", ":"))
                    prev_arguments = (
                        self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                        if self.current_tool_id < len(self.prev_tool_call_arr)
                        else None
                    )
                    prev_args_json = (
                        self._prev_args_jsons[self.current_tool_id]
                        if self.current_tool_id < len(self._prev_args_jsons)
                        else ""
                    )
                    argument_diff = None

                    if is_current_complete:
                        argument_diff = cur_args_json[sent:]
                        completing_tool_id = self.current_tool_id
                        self._buffer = current_text[start_idx + end_idx :]
                    elif prev_arguments is not None:
                        # Only re-serialize previous arguments if changed
                        if prev_args_json == "":
                            prev_args_json = json.dumps(
                                prev_arguments, separators=(",", ":")
                            )
                            if self.current_tool_id < len(self._prev_args_jsons):
                                self._prev_args_jsons[self.current_tool_id] = (
                                    prev_args_json
                                )
                        if cur_args_json != prev_args_json:
                            prefix = _find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = cur_args_json[len(prefix) :][sent:]

                    # Update prev_tool_call_arr with current state
                    if self.current_tool_id >= 0:
                        while len(self.prev_tool_call_arr) <= self.current_tool_id:
                            self.prev_tool_call_arr.append({})
                        self.prev_tool_call_arr[self.current_tool_id] = (
                            current_tool_call
                        )
                        # Update cached json
                        if self.current_tool_id >= len(self._prev_args_jsons):
                            self._prev_args_jsons += [""] * (
                                self.current_tool_id + 1 - len(self._prev_args_jsons)
                            )
                        self._prev_args_jsons[self.current_tool_id] = cur_args_json

                    if is_current_complete:
                        self.current_tool_name_sent = False
                        self.current_tool_id += 1

                    if argument_diff is not None and argument_diff:
                        tool_index_to_use = (
                            completing_tool_id
                            if is_current_complete
                            else self.current_tool_id
                        )
                        # Ensure streamed_args_for_tool is correct length
                        while len(self.streamed_args_for_tool) <= tool_index_to_use:
                            self.streamed_args_for_tool.append("")
                        self.streamed_args_for_tool[tool_index_to_use] += argument_diff

                        res = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=tool_index_to_use,
                                    parameters=argument_diff,
                                )
                            ],
                        )

            return res

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult()

    @abstractmethod
    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains function call markers specific to this format.
        """
        raise NotImplementedError()

    def supports_structural_tag(self) -> bool:
        """Return True if this detector supports structural tag format."""
        return True

    @abstractmethod
    def structure_info(self) -> _GetInfoFunc:
        """
        Return a function that creates StructureInfo for constrained generation.

        The returned function takes a tool name and returns a StructureInfo object
        containing the begin/end patterns and trigger tokens needed for constrained
        generation of function calls in this format.

        Returns:
            A function that takes a tool name (str) and returns StructureInfo
        """
        raise NotImplementedError()

    @abstractmethod
    def build_ebnf(self, tools: List[Tool]) -> str:
        """
        Build an EBNF grammar for constrained generation of function calls.

        This method generates an Extended Backus-Naur Form (EBNF) grammar that
        constrains the model's output to valid function calls in this format.
        The grammar should include all available tools and their parameter schemas.

        Args:
            tools: List of available tools/functions that can be called

        Returns:
            A string containing the EBNF grammar for this function call format

        The EBNF grammar should:
            - Define the overall structure of function calls in this format
            - Include all tool names from the provided tools list
            - Define valid JSON structures for function arguments
            - Handle multiple function calls if the format supports them

        Note:
            Most implementations use EBNFComposer.build_ebnf() utility with
            format-specific parameters rather than writing EBNF from scratch.
        """
        raise NotImplementedError()

    def _ensure_tool_indices(self, tools: List[Tool]):
        # Only rebuild index cache if tools set has changed (by id or length)
        if self._tools_ref is not tools:
            self._tool_indices = self._get_tool_indices(tools)
            self._tools_ref = tools
            # Invalidate argument JSON cache
            self._prev_args_jsons = ["" for _ in range(len(tools))]
