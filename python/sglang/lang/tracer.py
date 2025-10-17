"""Tracing a program."""

import uuid
from typing import Any, Dict, List, Optional

from sglang.lang.backend.base_backend import BaseBackend
from sglang.lang.interpreter import ProgramState, ProgramStateGroup
from sglang.lang.ir import (
    SglArgument,
    SglConstantText,
    SglExpr,
    SglExprList,
    SglFork,
    SglGen,
    SglGetForkItem,
    SglRoleBegin,
    SglRoleEnd,
    SglSelect,
    SglVariable,
    SglVarScopeBegin,
    SglVarScopeEnd,
)


class StopTracing(Exception):
    pass


def extract_prefix_by_tracing(program, backend):
    # Create dummy arguments
    dummy_arguments = {name: SglArgument(name, None) for name in program.arg_names}
    arguments = dummy_arguments
    arguments.update(program.bind_arguments)

    # Trace
    tracer = TracerProgramState(backend, arguments, only_trace_prefix=True)
    try:
        with TracingScope(tracer):
            tracer.ret_value = program.func(tracer, **arguments)
    except (StopTracing, TypeError, AttributeError):
        # Some exceptions may not be caught
        pass

    # Run and cache prefix
    prefix = ""
    for expr in tracer.flatten_nodes():
        if isinstance(expr, SglConstantText):
            prefix += expr.value
        else:
            break
    return prefix


def trace_program(program, arguments, backend):
    # Create dummy backend
    if backend is None:
        backend = BaseBackend()

    # Create dummy arguments
    dummy_arguments = {
        name: SglArgument(name, None)
        for name in program.arg_names
        if name not in arguments
    }
    arguments.update(dummy_arguments)
    arguments.update(program.bind_arguments)

    # Trace
    tracer = TracerProgramState(backend, arguments, only_trace_prefix=False)
    with TracingScope(tracer):
        tracer.ret_value = program.func(tracer, **arguments)
    return tracer


class TracerProgramState(ProgramState):
    def __init__(self, backend, arguments, only_trace_prefix):
        self.pid = uuid.uuid4().hex
        self.backend = backend
        self.arguments: Dict[str, Any] = arguments
        self.only_trace_prefix = only_trace_prefix

        if hasattr(backend, "endpoint"):
            self.backend = backend.endpoint

        self.nodes = []
        self.last_node = None
        self.variables = {}
        self.ret_value = None

        # For completion

        # For chat
        self.messages_ = []
        self.cur_role = None
        self.chat_template = self.backend.get_chat_template()

        # For multi states
        self.child_states = []

        cur_scope = TracingScope.get_current_scope()
        if cur_scope is not None:
            cur_scope.add_child_state(self)

    ##################################
    ########### Public API ###########
    ##################################

    def fork(self, size: int = 1, position_ids_offset: Optional[List[int]] = None):
        assert size >= 1

        if self.only_trace_prefix:
            raise StopTracing()

        fork_node = SglFork(size)
        fork_node.prev_node = self.last_node

        states = [
            TracerProgramState(self.backend, self.arguments, self.only_trace_prefix)
            for _ in range(size)
        ]

        for i in range(size):
            node = SglGetForkItem(i)
            node.prev_node = fork_node
            states[i].last_node = node
            states[i].variables = dict(self.variables)
            states[i].messages_ = list(self.messages_)
            states[i].cur_role = self.cur_role
            states[i].chat_template = self.chat_template

        state_group = ProgramStateGroup(states, self)

        return state_group

    ##################################
    ########## Internal API ##########
    ##################################

    def _append_node(self, other: SglExpr):
        self.nodes.append(other)
        # Store previous node only if there was one; attribute setting is faster if done conditionally
        other.prev_node = self.last_node
        self.last_node = other

    def _execute(self, other: SglExpr):
        # Avoid repeated type checks by using local variables and elif chain reordered by expected frequency
        # Directly use isinstance instead of checking str twice in _execute_fill
        # Also avoid attribute access on SglExpr subclasses unless required
        if isinstance(other, str):
            sct = SglConstantText(other)
            sct.pid = self.pid
            self._execute_fill(sct)
            return self

        other.pid = self.pid

        t = type(other)
        if t is SglConstantText:
            self._execute_fill(other)
        elif t is SglGen:
            self._execute_gen(other)
        elif t is SglSelect:
            self._execute_select(other)
        elif t is SglExprList:
            # Use local variable for improved PERF and micro-opt
            expr_list = other.expr_list
            for x in expr_list:
                self._execute(x)
        elif t is SglRoleBegin:
            self._execute_role_begin(other)
        elif t is SglRoleEnd:
            self._execute_role_end(other)
        elif t is SglVarScopeBegin:
            self._execute_var_scope_begin(other)
        elif t is SglVarScopeEnd:
            self._execute_var_scope_end(other)
        else:
            if self.only_trace_prefix:
                raise StopTracing()
            else:
                self._append_node(other)

        return self

    def __iadd__(self, other):
        self._execute(other)
        return self

    def _execute_fill(self, expr: SglConstantText):
        # No need to check isinstance(expr, str) because SglConstantText enforced above
        self._append_node(expr)

    def _execute_gen(self, expr: SglGen):
        # Use dict assignment only once and avoid redundant str()
        variables = self.variables
        name = expr.name
        if name is None:
            name = f"gen_{len(variables)}"
        new_node = SglVariable(name, source=expr)
        variables[name] = new_node
        self._append_node(expr)

    def _execute_select(self, expr: SglSelect):
        variables = self.variables
        name = expr.name
        if name is None:
            name = f"select_{len(variables)}"
        new_node = SglVariable(name, source=expr)
        variables[name] = new_node
        self._append_node(expr)

    def _execute_role_begin(self, expr: SglRoleBegin):
        assert self.cur_role is None, "Nested roles are not allowed."
        # Use locals for members for faster access
        messages_ = self.messages_
        chat_template = self.chat_template

        if not messages_ and expr.role != "system":
            default_system = chat_template.default_system_prompt
            if default_system:
                # Reuse these methods directly to avoid function call stack growth
                self._execute_role_begin(SglRoleBegin("system"))
                self._execute_fill(SglConstantText(default_system))
                self._execute_role_end(SglRoleEnd("system"))

        self.cur_role = expr.role

        prefix, suffix = chat_template.get_prefix_and_suffix(
            expr.role, messages_
        )
        self._execute_fill(SglConstantText(prefix))

    def _execute_role_end(self, expr: SglRoleEnd):
        messages_ = self.messages_
        chat_template = self.chat_template

        prefix, suffix = chat_template.get_prefix_and_suffix(
            expr.role, messages_
        )
        self._execute_fill(SglConstantText(suffix))

        # Use append with pre-built dict for speed
        messages_.append({"role": expr.role, "content": ""})

        self.cur_role = None

    def _execute_var_scope_end(self, expr: SglVarScopeEnd):
        # Use last_node directly
        self.variables[expr.name] = SglVariable(expr.name, source=self.last_node)

    def get_var(self, name):
        ret = self.arguments.get(name, None)
        if ret is not None:
            return ret

        v = self.variables[name]
        return SglVariable(v.name, v.source)

    def flatten_nodes(self):
        def traverse(cur):
            if isinstance(cur, SglExprList):
                for child in cur.expr_list:
                    traverse(child)
            else:
                ret.append(cur)

        ret = []
        for x in self.nodes:
            traverse(x)
        return ret

    def __del__(self):
        pass


class TracingScope:
    cur_scope = None

    def __init__(self, tracer_state: TracerProgramState):
        self.tracer_state = tracer_state
        self.last_scope = TracingScope.cur_scope

    def __enter__(self):
        TracingScope.cur_scope = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        TracingScope.cur_scope = self.last_scope

    @staticmethod
    def get_current_scope():
        return TracingScope.cur_scope

    def add_child_state(self, state: TracerProgramState):
        cur_scope = self
        while cur_scope is not None:
            cur_scope.tracer_state.child_states.append(state)
            cur_scope = cur_scope.last_scope
