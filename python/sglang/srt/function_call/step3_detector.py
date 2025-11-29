import ast
import json
import logging
import re
from typing import Any, Dict, List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


def get_argument_type(func_name: str, arg_key: str, defined_tools: List[Tool]) -> str:
    """Get the expected type for a function argument from tool schema."""
    name2tool = {tool.function.name: tool for tool in defined_tools}
    if func_name not in name2tool:
        return None
    tool = name2tool[func_name]
    parameters = tool.function.parameters or {}
    properties = parameters.get("properties", {})
    if arg_key not in properties:
        return None
    return properties[arg_key].get("type", None)


def parse_arguments(value: str) -> tuple[Any, bool]:
    """Parse a string value to appropriate type. Returns (parsed_value, success)."""
    try:
        try:
            parsed_value = json.loads(value)
        except:
            parsed_value = ast.literal_eval(value)
        return parsed_value, True
    except:
        return value, False


class Step3Detector(BaseFormatDetector):
    """
    Detector for Step3 model function call format.

    The Step3 format uses special Unicode tokens to delimit function calls
    with steptml XML format for invocations.

    Format Structure:
    ```
    <｜tool_calls_begin｜>
    <｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="function_name">
    <steptml:parameter name="param1">value1</steptml:parameter>
    <steptml:parameter name="param2">value2</steptml:parameter>
    </steptml:invoke><｜tool_call_end｜>
    <｜tool_calls_end｜>
    ```
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜tool_calls_begin｜>"
        self.eot_token = "<｜tool_calls_end｜>"
        self.tool_call_begin = "<｜tool_call_begin｜>"
        self.tool_call_end = "<｜tool_call_end｜>"
        self.tool_sep = "<｜tool_sep｜>"

        # Regex for parsing steptml invocations
        self.invoke_regex = re.compile(
            r'<steptml:invoke name="([^"]+)">(.+?)</steptml:invoke>', re.DOTALL
        )
        self.param_regex = re.compile(
            r'<steptml:parameter name="([^"]+)">([^<]*)</steptml:parameter>', re.DOTALL
        )

        # Streaming state variables
        self._in_tool_block: bool = False
        self._tool_block_finished: bool = False
        self._current_function_name: str = ""
        self._current_parameters: Dict[str, Any] = {}
        self._in_tool_call: bool = False
        self._function_name_sent: bool = False

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Step3 format tool call."""
        return self.bot_token in text

    def _parse_steptml_invoke(
        self, text: str, tools: List[Tool] = None
    ) -> tuple[str, dict]:
        """Parse steptml invoke format to extract function name and parameters."""
        invoke_match = self.invoke_regex.search(text)
        if not invoke_match:
            return None, {}

        func_name = invoke_match.group(1)
        params_text = invoke_match.group(2)

        params = {}
        for param_match in self.param_regex.finditer(params_text):
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()

            # If tools provided, use schema-aware parsing
            if tools:
                arg_type = get_argument_type(func_name, param_name, tools)
                if arg_type and arg_type != "string":
                    parsed_value, _ = parse_arguments(param_value)
                    params[param_name] = parsed_value
                else:
                    params[param_name] = param_value
            else:
                # Fallback to generic parsing if no tools provided
                parsed_value, _ = parse_arguments(param_value)
                params[param_name] = parsed_value

        return func_name, params

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.
        """
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])

        try:
            pre_text, rest = text.split(self.bot_token, 1)

            # If no end token, return everything as normal text
            if self.eot_token not in rest:
                return StreamingParseResult(normal_text=text, calls=[])

            tool_section, post_text = rest.split(self.eot_token, 1)

            # Find all individual tool calls using regex
            calls = []
            tool_call_pattern = (
                f"{re.escape(self.tool_call_begin)}(.*?){re.escape(self.tool_call_end)}"
            )

            for match in re.finditer(tool_call_pattern, tool_section, re.DOTALL):
                call_content = match.group(1)

                # Check if it's a function call
                if self.tool_sep not in call_content:
                    continue

                type_part, invoke_part = call_content.split(self.tool_sep, 1)
                if type_part.strip() != "function":
                    continue

                func_name, params = self._parse_steptml_invoke(invoke_part, tools)
                if func_name:
                    # Use parse_base_json to create the ToolCallItem
                    action = {"name": func_name, "arguments": params}
                    calls.extend(self.parse_base_json(action, tools))

            # Combine pre and post text
            normal_text = pre_text + post_text

            return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # Return the original text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for Step3 format.
        """
        self._buffer += new_text

        # Build tool indices for validation (cache after first use)
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        # Early return if tool block is finished; avoid further parsing work
        if self._tool_block_finished:
            normal_text = self._buffer
            self._buffer = ""
            return StreamingParseResult(normal_text=normal_text)

        # Not started tool block yet
        if not self._in_tool_block:
            bot_token = self.bot_token
            buf = self._buffer
            idx = buf.find(bot_token)
            if idx != -1:
                # Fast path: found tool block start
                normal_text = buf[:idx]
                self._buffer = buf[idx + len(bot_token):]
                self._in_tool_block = True
                return StreamingParseResult(normal_text=normal_text)
            else:
                partial_len = self._ends_with_partial_token(
                    buf, bot_token
                )
                if partial_len:
                    return StreamingParseResult()
                else:
                    normal_text = buf
                    self._buffer = ""
                    return StreamingParseResult(normal_text=normal_text)

        # Inside the tool block
        calls: List[ToolCallItem] = []

        # Quick check for tool block end
        eot_token = self.eot_token
        buf = self._buffer
        eot_idx = buf.find(eot_token)
        if eot_idx != -1:
            # Handle possible end-of-tool call completion
            if self._in_tool_call:
                before_eot = buf[:eot_idx]
                if self.tool_call_end in before_eot:
                    result = self._parse_partial_tool_call(tools)
                    calls.extend(result.calls)
                # else: incomplete tool call; warning skipped for performance

            remaining = buf[eot_idx + len(eot_token):]
            self._buffer = ""
            self._tool_block_finished = True
            self._reset_streaming_state()
            return StreamingParseResult(normal_text=remaining, calls=calls)

        # Tool call state management (start new call or continue incremental parsing)
        if not self._in_tool_call:
            tc_begin = self.tool_call_begin
            idx = buf.find(tc_begin)
            if idx != -1:
                self._buffer = buf[idx + len(tc_begin):]
                self._in_tool_call = True
                self._function_name_sent = False
                self._current_function_name = ""
                self._current_parameters = {}
            else:
                return StreamingParseResult()

        # Always try to parse the partial tool call when in_call
        if self._in_tool_call:
            return self._parse_partial_tool_call(tools)

        return StreamingParseResult()

    def _parse_partial_tool_call(self, tools: List[Tool]) -> StreamingParseResult:
        """Parse partial tool call for streaming scenarios."""
        calls = []
        buf = self._buffer

        # Only proceed if we have tool_sep, i.e., we're past the declaration type field
        sep = self.tool_sep
        sep_idx = buf.find(sep)
        if sep_idx == -1:
            return StreamingParseResult(calls=calls)

        type_part = buf[:sep_idx]
        invoke_part = buf[sep_idx + len(sep):]
        if type_part.strip() != "function":
            self._reset_streaming_state()
            return StreamingParseResult(calls=calls)

        # Extract function name (streamed incrementally; skip work if already sent)
        if not self._function_name_sent:
            match = self.invoke_regex.search(invoke_part)
            if match:
                func_name = match.group(1)
                if func_name in self._tool_indices:
                    self._current_function_name = func_name
                    self._function_name_sent = True
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                    # Expand tracking arrays just enough
                    ptca = self.prev_tool_call_arr
                    satf = self.streamed_args_for_tool
                    cur_tid = self.current_tool_id
                    if len(ptca) <= cur_tid:
                        ptca.extend({} for _ in range(cur_tid + 1 - len(ptca)))
                    if len(satf) <= cur_tid:
                        satf.extend("" for _ in range(cur_tid + 1 - len(satf)))
                    ptca[cur_tid] = {"name": func_name, "arguments": {}}
                    calls.append(
                        ToolCallItem(
                            tool_index=cur_tid,
                            name=func_name,
                            parameters="",
                        )
                    )
                # else: silently ignore invalid function names for streaming speed
            else:
                return StreamingParseResult(calls=calls)

        # Parameters parsing: extract and compare only if there may be a change
        param_matches = list(self.param_regex.finditer(invoke_part))
        if self._function_name_sent:
            new_params = {}
            for pm in param_matches:
                param_name = pm.group(1)
                param_value = pm.group(2).strip()
                # Get expected arg type
                arg_type = get_argument_type(
                    self._current_function_name, param_name, tools
                )
                # Only call parse_arguments for non-string types
                if arg_type and arg_type != "string":
                    parsed_value, _ = parse_arguments(param_value)
                    new_params[param_name] = parsed_value
                else:
                    new_params[param_name] = param_value

            # Efficient diff: check keys first
            cur_params = self._current_parameters
            if new_params != cur_params:
                # Stream-by-JSON: send incremental JSON diff
                if not cur_params:
                    # First parameter(s): send opening brace + values
                    params_content = json.dumps(new_params, ensure_ascii=False)
                    if len(params_content) > 2:
                        diff = params_content[:-1]
                    else:
                        diff = "{"
                else:
                    old_json = json.dumps(cur_params, ensure_ascii=False)
                    new_json = json.dumps(new_params, ensure_ascii=False)
                    # Skip closing braces
                    old_no_brace = old_json[:-1]
                    new_no_brace = new_json[:-1]
                    if new_no_brace.startswith(old_no_brace):
                        diff = new_no_brace[len(old_no_brace):]
                    else:
                        diff = ""
                if diff:
                    ctidx = self.current_tool_id
                    calls.append(
                        ToolCallItem(
                            tool_index=ctidx,
                            parameters=diff,
                        )
                    )
                    self.streamed_args_for_tool[ctidx] += diff
                # State update
                self._current_parameters = new_params
                self.prev_tool_call_arr[self.current_tool_id]["arguments"] = new_params

            # End-of-call handling (send closing brace if any params streamed)
            tc_end = self.tool_call_end
            end_idx = buf.find(tc_end)
            if end_idx != -1:
                ctidx = self.current_tool_id
                if self.streamed_args_for_tool[ctidx]:
                    calls.append(
                        ToolCallItem(
                            tool_index=ctidx,
                            parameters="}",
                        )
                    )
                    self.streamed_args_for_tool[ctidx] += "}"
                self._buffer = buf[end_idx + len(tc_end):]
                self._reset_streaming_state()
                self.current_tool_id += 1

        return StreamingParseResult(calls=calls)

    def _reset_streaming_state(self):
        """Reset streaming state for the next tool call"""
        self._in_tool_call = False
        self._function_name_sent = False
        self._current_function_name = ""
        self._current_parameters = {}

    def supports_structural_tag(self) -> bool:
        """Return True if this detector supports structural tag format."""
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()
