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

        # Pre-cache references for tighter loops
        get_value_rule = EBNFComposer.get_value_rule

        # =================================================================
        # Step 4: Build rules for each tool
        # =================================================================
        for tool in tools:
            tool_name = tool.function.name
            params = tool.function.parameters or {}
            properties = params.get("properties", {})
            required_props = set(params.get("required", ()))

            prop_kv_pairs = {}
            ordered_props = list(properties.keys())
            # Precompute key-value pairs in a single loop for order efficiency
            for prop_name in ordered_props:
                pair = key_value_template.format(
                    key=prop_name, valrule=get_value_rule(properties[prop_name], function_format)
                )
                prop_kv_pairs[prop_name] = pair

            required = []
            optional = []
            # Avoid repeated lookups and keep order
            for prop in ordered_props:
                if prop in required_props:
                    required.append(prop)
                else:
                    optional.append(prop)

            rule_parts = []

            # Add required properties joined by commas
            if required:
                # Use list comprehension for join efficiency
                rule_parts.append(
                    f" {key_value_separator} ".join(prop_kv_pairs[k] for k in required)
                )

            # Add optional properties with flexible ordering
            if optional:
                # Prebuild alternatives in one loop
                opt_alternatives = []
                n_opt = len(optional)
                for i in range(n_opt):
                    # Use preallocated list and string joins for minimized temporaries
                    opt_parts = [prop_kv_pairs[optional[i]]]
                    for j in range(i + 1, n_opt):
                        opt_parts.append(
                            f" ( {key_value_separator} {prop_kv_pairs[optional[j]]} )?"
                        )
                    opt_alternatives.append("".join(opt_parts))

                # Only keep one string join
                opt_block = " | ".join(opt_alternatives)
                if required:
                    rule_parts.append(f" ( {key_value_separator} ( {opt_block} ) )?")
                else:
                    rule_parts.append(f"( {opt_block} )?")

            combined_args = "".join(rule_parts)
            arguments_rule = args_template.format(arg_rules=combined_args)
            if not arguments_rule:
                arguments_rule = '""'

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
        # Avoid redundant dictionary lookup
        base_grammar = grammar_dict.get(
            function_format, EBNFComposer.json_grammar_ebnf_str
        )
        ebnf_lines.append(base_grammar)

        return "\n".join(ebnf_lines)
