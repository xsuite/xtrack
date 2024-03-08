import builtins
import functools

import numpy as np
from cython.cimports.libc.stdio import fopen  # noqa
from cython.cimports.posix.unistd import access, R_OK  # noqa
from cython.cimports.xtrack.xmad import xmad  # noqa

from typing import List
import operator
import math

from xdeps.refs import CallRef, LiteralExpr


BUILTIN_FUNCTIONS = {
    name: getattr(math, name) for name in [
        'sqrt', 'log', 'log10', 'exp', 'sin', 'cos', 'tan',
        'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
        'erf', 'erfc', 'floor', 'ceil',
    ]
}
BUILTIN_FUNCTIONS["sinc"] = np.sinc
BUILTIN_FUNCTIONS["abs"] = builtins.abs
BUILTIN_FUNCTIONS["round"] = builtins.round
BUILTIN_FUNCTIONS["frac"] = lambda x: x % 1.0


class ParseLogEntry:
    def __init__(self, location, reason, context=None):
        self.line_no = location['first_line']
        self.column = location['first_column']
        self.end_line_no = location['last_line']
        self.end_column = location['last_column']
        self.reason = reason
        self.context = context

    def __repr__(self):
        out = f'On line {self.line_no} column {self.column}: {self.reason}'
        if self.context:
            out += '\n----> ' + self.context
        return out


parse_log: List[ParseLogEntry] = []  # not thread safe


class XMadParseError(Exception):
    def __init__(self, parse_log):
        self.parse_log = parse_log.copy()
        message = '\n\n'.join([repr(entry) for entry in parse_log])
        super().__init__(message)


def yyerror(message):
    message_str = message.decode()
    text = xmad.yytext.decode()
    parse_log.append(ParseLogEntry(xmad.yylloc, message_str, context=text))


def parse_file(path):
    global parse_log
    parse_log = []

    c_path = path.encode()

    file_readable = access(c_path, R_OK)
    if file_readable != 0:
        raise OSError(f'File {path} does not exist or is not readable.')

    xmad.yyin = fopen(c_path, "r")

    success = xmad.yyparse()
    xmad.yylex_destroy()

    if success != 0 or parse_log:
        raise XMadParseError(parse_log)


def parse_string(string):
    global parse_log
    parse_log = []

    xmad.yy_scan_string(string.encode())
    success = xmad.yyparse()
    xmad.yylex_destroy()

    if success != 0 or parse_log:
        raise XMadParseError(parse_log)


def py_float(value):
    return LiteralExpr(value)


def py_unary_op(op_string, value):
    function = getattr(operator, op_string.decode())
    return function(value)


def py_binary_op(op_string, left, right):
    function = getattr(operator, op_string.decode())
    return function(left, right)


def py_call_func(func_name, value):
    normalized_name = func_name.decode().lower()
    if normalized_name not in BUILTIN_FUNCTIONS:
        yyerror(f'builtin function `{normalized_name}` is unknown'.encode())
        return None

    function = BUILTIN_FUNCTIONS[normalized_name]
    return CallRef(function, (value,), {})


def py_eq_value_scalar(identifier, value):
    print(f'==> {identifier} = {value}')
    return identifier, value


def py_eq_defer_scalar(identifier, value):
    print(f'==> {identifier} := {value}')
    return identifier, value