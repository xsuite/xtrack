# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Bind text calculations to the same runtime as numerical callbacks."""

import ast
from numbers import Number
from types import SimpleNamespace

from .timing import _CalculationTiming


# Arity excludes output arrays and backend-specific metadata. The only optional
# argument is a literal axis for reductions, stacking and finite differences.
_OPERATIONS = {
    'mean': (1, True),
    'sum': (1, True),
    'stack': (1, True),
    'diff': (1, True),
    'sin': (1, False),
    'maximum': (2, False),
    'where': (3, False),
    'linalg.solve': (2, False),
}
_OPERATION_ROOTS = {name.split('.')[0] for name in _OPERATIONS}
_BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.MatMult,
           ast.BitAnd, ast.BitOr)
_UNARY = (ast.UAdd, ast.USub, ast.Invert)
_COMPARE = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


def _integer_literal(node):
    """Accept a host index or axis without evaluating a numerical array."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        node = node.operand
    return isinstance(node, ast.Constant) and type(node.value) is int


def _backend_operation(xp, root, member):
    """Bind a native operation with consistent scalar and error handling."""
    function = getattr(xp, root)
    if member:
        function = getattr(function, member)
    if root == 'stack':
        def stack(values, *, axis=0):
            return function(tuple(xp.asarray(value) for value in values), axis=axis)
        return stack
    if root in ('mean', 'sum', 'diff', 'where'):
        # CuPy requires an array for reductions and the where predicate.
        def on_array(value, *args, **kwargs):
            return function(xp.asarray(value), *args, **kwargs)
        return on_array
    if root == 'linalg' and member == 'solve' and xp.__name__ == 'cupy':
        from cupyx import errstate

        def solve(matrix, rhs):
            # Check cuSOLVER status before it can produce an applied command.
            with errstate(linalg='raise'):
                return function(matrix, rhs)
        return solve
    return function


class _Program:
    """Validate a placed assignment block and bind its native array operations.

    Python evaluates the admitted expressions. Only locals and explicitly
    bound numerical names are available. Retained state and physical commands
    are validated and committed by the controller.
    """

    def __init__(self, source, *, name, acquired, parameters, state, output, xp):
        self._name = name
        self._bindings = {
            'acquired': set(acquired),
            'parameters': set(parameters),
            'state': set(state),
            'outputs': {output},
            'event': set(_CalculationTiming.__annotations__),
        }
        self._locals = set()
        self._output_assigned = False
        self._globals = {'__builtins__': {}}
        self._xp = xp
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            self._error(error, error.msg)
        for statement in tree.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                self._error(statement, 'Only numerical assignments are supported')
            self._value(statement.value)
            self._target(statement.targets[0])
        if not self._output_assigned:
            self._error(tree, f'Assign the bound output {output!r}')
        self._code = compile(tree, f'<feedback {name}>', 'exec', dont_inherit=True)

    def _error(self, node, message):
        line = getattr(node, 'lineno', None) or 1
        raise ValueError(f'Feedback {self._name!r}, program line {line}: {message}')

    def _field(self, node, namespaces):
        if (
            not isinstance(node, ast.Attribute)
            or not isinstance(node.value, ast.Name)
            or node.value.id not in namespaces
            or node.attr.startswith('_')
        ):
            self._error(node, 'Use a bound numerical name')
        namespace = node.value.id
        if node.attr not in self._bindings[namespace]:
            self._error(node, f'Unknown {namespace} field {node.attr!r}')
        return namespace

    def _target(self, node):
        if isinstance(node, ast.Name):
            if (
                node.id in self._bindings or node.id in _OPERATION_ROOTS
                or node.id.startswith('_')
            ):
                self._error(node, f'Reserved assignment name {node.id!r}')
            self._locals.add(node.id)
        else:
            namespace = self._field(node, {'state', 'outputs'})
            if namespace == 'outputs':
                self._output_assigned = True

    def _value(self, node):
        if isinstance(node, ast.Name):
            if node.id not in self._locals:
                self._error(node, f'Unknown local name {node.id!r}')
        elif isinstance(node, ast.Constant) and isinstance(node.value, Number):
            return
        elif isinstance(node, ast.Attribute):
            namespace = self._field(node, self._bindings)
            if namespace == 'outputs' and not self._output_assigned:
                self._error(node, f'Output {node.attr!r} has not been assigned')
        elif isinstance(node, ast.Call):
            self._call(node)
        elif isinstance(node, ast.Subscript):
            self._value(node.value)
            indices = (
                node.slice.elts if isinstance(node.slice, ast.Tuple)
                else (node.slice,)
            )
            for index in indices:
                bounds = (
                    (index.lower, index.upper, index.step)
                    if isinstance(index, ast.Slice) else (index,)
                )
                if any(
                    bound is not None and not _integer_literal(bound)
                    for bound in bounds
                ):
                    self._error(index, 'Use integer literals for indices and slices')
                if (
                    isinstance(index, ast.Slice) and index.step is not None
                    and ast.literal_eval(index.step) == 0
                ):
                    self._error(index, 'Slice step must be nonzero')
        elif isinstance(node, ast.BinOp) and isinstance(node.op, _BINARY):
            self._value(node.left)
            self._value(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, _UNARY):
            self._value(node.operand)
        elif (
            isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], _COMPARE)
        ):
            self._value(node.left)
            self._value(node.comparators[0])
        else:
            self._error(node, f'Unsupported expression {type(node).__name__}')

    def _call(self, node):
        target = node.func
        if isinstance(target, ast.Name):
            name = target.id
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            name = f'{target.value.id}.{target.attr}'
        else:
            name = ''
        if name not in _OPERATIONS:
            self._error(node, f'Unsupported operation {ast.unparse(target)!r}')
        arity, allows_axis = _OPERATIONS[name]
        if len(node.args) != arity:
            self._error(node, f'{name} requires {arity} positional arguments')
        values = node.args
        if name == 'stack' and isinstance(node.args[0], (ast.Tuple, ast.List)):
            # Stack the context-native values rather than converting a list.
            values = node.args[0].elts
            if not values:
                self._error(node, 'stack requires at least one value')
        for value in values:
            self._value(value)
        if node.keywords:
            if not allows_axis:
                self._error(node, f'{name} does not accept keyword arguments')
            option = node.keywords[0]
            if not (
                len(node.keywords) == 1 and option.arg == 'axis'
                and _integer_literal(option.value)
            ):
                self._error(node, f'{name} accepts only a literal axis')
        root, _, member = name.partition('.')
        scope = self._globals
        if member:
            scope = vars(self._globals.setdefault(root, SimpleNamespace()))
        key = member or root
        if key not in scope:
            try:
                scope[key] = _backend_operation(self._xp, root, member)
            except AttributeError:
                self._error(node, f'{name} is unavailable on {self._xp.__name__}')

    def __call__(self, inputs, parameters, previous, event):
        """Stage the calculation with the controller's context-native arrays."""
        state = SimpleNamespace(**previous)
        outputs = SimpleNamespace()
        scope = {
            'acquired': SimpleNamespace(**inputs),
            'parameters': SimpleNamespace(**parameters),
            'state': state,
            'outputs': outputs,
            'event': SimpleNamespace(**{
                name: event[name] for name in _CalculationTiming.__annotations__
            }),
        }
        try:
            exec(self._code, self._globals, scope)
        except Exception as error:
            line = error.__traceback__.tb_next.tb_lineno
            raise ValueError(
                f'Feedback {self._name!r}, program line {line}: {error}'
            ) from error
        return {'memory': vars(state), 'outputs': vars(outputs)}
