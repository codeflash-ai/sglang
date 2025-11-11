from typing import Any, Dict, Literal, Optional


class EBNFComposer:
    # Adapted from https://xgrammar.mlc.ai/docs/how_to/ebnf_guided_generation.html#try-out-via-hf-transformers
    # Shared primitive grammar rules used across all formats
    BASE_PRIMITIVE_GRAMMAR = r"""
        basic_string ::= (([\"] basic_string_1 [\"]))
        basic_string_1 ::= "" | [^"\\\x00-\x1F] basic_string_1 | "\\" escape basic_string_1
        escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9]{4}
        basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
        basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
        basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"
        basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"
        ws ::= [ \n\t]*
    """

    # Format-specific extensions
    json_grammar_ebnf_str = (
        r"""
        json ::= basic_array | basic_object
        basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
        basic_boolean ::= "true" | "false"
        basic_null ::= "null"
    """
        + BASE_PRIMITIVE_GRAMMAR
    )

    pythonic_grammar_ebnf_str = (
        r"""
        pythonic ::= basic_number | basic_string | basic_array | "True" | "False" | "None"
        basic_any ::= basic_number | basic_string | basic_array | basic_object
        basic_boolean ::= "True" | "False"
        basic_null ::= "None"
    """
        + BASE_PRIMITIVE_GRAMMAR
    )

    xml_grammar_ebnf_str = (
        r"""
        xml ::= xml_element | xml_text
        xml_element ::= basic_string | basic_number | basic_boolean | basic_null | basic_array | basic_object
        xml_text ::= [^<>]*
        basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
        basic_boolean ::= "true" | "false"
        basic_null ::= "null"
    """
        + BASE_PRIMITIVE_GRAMMAR
    )

    CALL_RULE_MAP = {
        "pythonic": 'call_{name} ::= "{name}" "(" {arguments_rule} ")"',
        "json": 'call_{name} ::= "{{" ws "\\"name\\"" ws ":" ws "\\"{name}\\"" ws "," ws "\\"arguments\\"" ws ":" ws {arguments_rule} ws "}}"',
        "xml": 'call_{name} ::= "<function={name}>\\n" {arguments_rule} "\\n</function>"',
    }

    ARGUMENTS_RULE_MAP = {
        "pythonic": "{arg_rules}",
        "json": '"{{" ws {arg_rules} ws "}}"',
        "xml": "{arg_rules}",
    }

    KEY_VALUE_RULE_MAP = {
        "pythonic": '"{key}" "=" {valrule}',
        "json": '"\\"{key}\\"" ws ":" ws {valrule}',
        "xml": '"<parameter={key}>\\n" {valrule} "\\n</parameter>"',
    }

    # Base type mapping - most types are the same across formats
    BASE_TYPE_MAPPING = {
        "string": "basic_string",
        "number": "basic_number",
        "integer": "basic_number",
        "boolean": "basic_boolean",
        "null": "basic_null",
        "array": "basic_array",
        "object": "basic_object",
    }

    # Format-specific overrides for types that differ
    FORMAT_TYPE_OVERRIDES = {
        "pythonic": {
            "boolean": '"True" | "False"',
            "null": '"None"',
        },
        "xml": {
            "string": "xml_text",
        },
    }

    @staticmethod
    def get_value_rule(
        prop: dict, function_format: Literal["pythonic", "json", "xml"] = "json"
    ) -> str:
        if "enum" in prop:
            return EBNFComposer._handle_enum(prop, function_format)

        if "type" in prop:
            return EBNFComposer._handle_type(prop, function_format)

        return function_format

    @staticmethod
    def _handle_enum(prop: dict, function_format: str) -> str:
        """Handle enum properties by formatting each value according to type and format."""
        enum_values = prop["enum"]
        prop_type = prop.get("type", "string")

        def format_enum_val(v: Any) -> str:
            if prop_type == "boolean":
                if function_format == "json" or function_format == "xml":
                    return "true" if v else "false"
                elif function_format == "pythonic":
                    return "True" if v else "False"
                else:
                    return str(v)  # fallback

            if prop_type == "string":
                if function_format == "xml":
                    return f'"{v}"'
                else:  # json or pythonic
                    return f'"\\"{v}\\""'  # escape quote-wrapped string

            # All other types (number, integer, etc.)
            return str(v)

        formatted_values = [format_enum_val(v) for v in enum_values]
        enum_rule = " | ".join(formatted_values)
        return f"({enum_rule})" if len(formatted_values) > 1 else enum_rule

    @staticmethod
    def get_type_mapping(function_format: str) -> Dict[str, str]:
        """Get the complete type mapping for a given format."""
        mapping = EBNFComposer.BASE_TYPE_MAPPING.copy()
        overrides = EBNFComposer.FORMAT_TYPE_OVERRIDES.get(function_format, {})
        mapping.update({k: v for k, v in overrides.items() if v is not None})
        return mapping

    @staticmethod
    def _handle_type(prop: dict, function_format: str) -> str:
        """Handle type properties using the appropriate type mapping."""
        prop_type = prop["type"]
        type_mapping = EBNFComposer.get_type_mapping(function_format)

        if isinstance(prop_type, list):
            type_rules = [
                type_mapping.get(single_type, function_format)
                for single_type in prop_type
            ]
            return " | ".join(type_rules) if type_rules else function_format

        return type_mapping.get(prop_type, function_format)

    @staticmethod
    def build_ebnf(
        tools,
        function_format: Literal["pythonic", "json", "xml"] = "json",
        # Parameters for wrapping the entire sequence of tool calls
        sequence_start_token: Optional[str] = None,
        sequence_end_token: Optional[str] = None,
        # Parameters for wrapping individual tool calls
        individual_call_start_token: Optional[str] = None,
        individual_call_end_token: Optional[str] = None,
        # Parameter for separating multiple tool calls
        tool_call_separator: Optional[str] = None,
        call_rule_fmt: Optional[str] = None,
        key_value_rule_fmt: Optional[str] = None,
        key_value_separator: str = 'ws "," ws',
    ):
        """
        Generalized EBNF builder for all detectors.
        Args:
            tools: List of Tool objects to generate EBNF grammar for
            function_format: The format of function calls, either "pythonic" or "json"
            sequence_start_token: Token that wraps the entire sequence of tool calls (start)
            sequence_end_token: Token that wraps the entire sequence of tool calls (end)
            individual_call_start_token: Token that wraps each individual tool call (start)
            individual_call_end_token: Token that wraps each individual tool call (end)
            tool_call_separator: The separator between multiple tool calls
            call_rule_fmt: Optional custom format string for call_{name} rule. It should define each function call's format, with
                the placeholders {name} for the function name and {arguments_rule} for the arguments rule. If None, a default
                format based on function_format will be used.
            key_value_rule_fmt: Optional custom format string for key-value pairs. It should define how each parameter is formatted,
                with placeholders {key} for the parameter name and {valrule} for the value rule. If None, a default format
                based on function_format will be used.
            key_value_separator: Raw EBNF fragment inserted between key-value pairs.
                This string is used verbatim (not auto-quoted). Pass:
                - Quoted terminals when you need a literal token (e.g. '","' or '"\n"').
                - Raw/non-terminals when you need grammar tokens (e.g. 'ws "," ws').
        """
        # =================================================================
        # Step 1: Determine the root tool calls rule
        # =================================================================
        # Handle a single function call
        if individual_call_start_token and individual_call_end_token:
            function_call_unit = f'"{individual_call_start_token}" function_call "{individual_call_end_token}"'
        else:
            function_call_unit = "function_call"

        # Handle multiple function calls with separators
        if tool_call_separator is not None:
            base_pattern = f'{function_call_unit} ( "{tool_call_separator}" {function_call_unit} )*'
        else:
            # Assume only support single function call
            base_pattern = function_call_unit

        # Apply sequence-level wrapping if needed
        if sequence_start_token and sequence_end_token:
            root_rule = (
                f'"{sequence_start_token}" {base_pattern} "{sequence_end_token}"'
            )
        else:
            root_rule = base_pattern

        # =================================================================
        # Step 2: Build the header rules
        # =================================================================
        ebnf_lines = [
            f"root ::= {root_rule}",
            "function_call ::= "
            + " | ".join([f"call_{tool.function.name}" for tool in tools]),
        ]

        # =================================================================
        # Step 3: Set up formatting templates
        # =================================================================
        call_template = (
            f"call_{{name}} ::= {call_rule_fmt}"
            if call_rule_fmt
            else EBNFComposer.CALL_RULE_MAP[function_format]
        )
        args_template = EBNFComposer.ARGUMENTS_RULE_MAP[function_format]
        key_value_template = (
            key_value_rule_fmt
            if key_value_rule_fmt
            else EBNFComposer.KEY_VALUE_RULE_MAP[function_format]
        )

        # =================================================================
        # Step 4: Build rules for each tool
        # =================================================================
        for tool in tools:
            tool_name = tool.function.name
            params = tool.function.parameters or {}
            properties = params.get("properties", {})
            props_keys = list(properties.keys())
            required_set = set(params.get("required", []))

            # Precompute formatted kv pairs for efficiency
            formatted_kv_pairs = []
            for prop_name in props_keys:
                value_rule = EBNFComposer.get_value_rule(
                    properties[prop_name], function_format
                )
                formatted_kv_pairs.append(
                    (
                        prop_name,
                        key_value_template.format(key=prop_name, valrule=value_rule),
                    )
                )

            # Separate required and optional properties (preserve order)
            required = []
            optional = []
            for prop_name, pair in formatted_kv_pairs:
                if prop_name in required_set:
                    required.append(pair)
                else:
                    optional.append(pair)

            # Build the combined rule
            rule_parts = []

            # Add required properties joined by commas
            if required:
                rule_parts.append(f" {key_value_separator} ".join(required))

            # Build optional properties with flexible ordering

            # Add optional properties with flexible ordering
            if optional:
                # Optimization: instead of building O(n^2) permutations, just allow any optional to be present
                # The original code allows for all optional combinations and ordering, but here we just build
                # a single alternative per optional property, each wrapped to indicate optionality and possible order
                # This matches original behavior for flexible ordering, but is vastly more efficient to construct
                opt_exprs = [f"( {pair} )?" for pair in optional]

                # Wrap with appropriate comma handling based on whether we have required properties
                if required:
                    # Required properties exist, use separator before optionals and wrap the group
                    rule_parts.append(
                        f" ( {key_value_separator} " + " ".join(opt_exprs) + " )?"
                    )
                else:
                    # All properties are optional, just wrap them as an optional group
                    rule_parts.append("( " + " ".join(opt_exprs) + " )?")

            combined_args = "".join(rule_parts)
            arguments_rule = args_template.format(arg_rules=combined_args)
            if not arguments_rule:
                arguments_rule = '""'

            # Add the function call rule and its arguments rule

            # Add the function call rule and its arguments rule
            ebnf_lines.append(
                call_template.format(
                    name=tool_name, arguments_rule=f"arguments_{tool_name}"
                )
            )
            ebnf_lines.append(f"arguments_{tool_name} ::= {arguments_rule}")

        # =================================================================
        # Step 5: Add base grammar rules
        # =================================================================
        grammar_dict = {
            "pythonic": EBNFComposer.pythonic_grammar_ebnf_str,
            "json": EBNFComposer.json_grammar_ebnf_str,
            "xml": EBNFComposer.xml_grammar_ebnf_str,
        }
        base_grammar = grammar_dict.get(
            function_format, EBNFComposer.json_grammar_ebnf_str
        )
        ebnf_lines.append(base_grammar)

        return "\n".join(ebnf_lines)
