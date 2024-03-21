import builtins
import contextlib
from typing import Tuple

import cython as cy
import numpy as np
import scipy.constants as sc
from cython.cimports.libc.stdio import fopen  # noqa
from cython.cimports.posix.unistd import access, R_OK  # noqa
from cython.cimports.xtrack.xmad import xmad  # noqa
from cython.cimports.cpython.ref import Py_INCREF, PyObject  # noqa

import math
import operator
import traceback

import xdeps as xd
import xtrack as xt
from xdeps.refs import CallRef, LiteralExpr
from xtrack.beam_elements import element_classes as xt_element_classes


KEEP_LITERAL_EXPRESSIONS = False

BUILTIN_FUNCTIONS = {
    name: getattr(math, name) for name in [
        'sqrt', 'log', 'log10', 'exp', 'sin', 'cos', 'tan', 'asin', 'acos',
        'atan', 'sinh', 'cosh', 'tanh', 'erf', 'erfc',
    ]
}
BUILTIN_FUNCTIONS['sinc'] = np.sinc
BUILTIN_FUNCTIONS['abs'] = builtins.abs
BUILTIN_FUNCTIONS['round'] = np.round  # these can also be builtins, but the
BUILTIN_FUNCTIONS['floor'] = np.floor  # numpy versions take NaNs, which is
BUILTIN_FUNCTIONS['ceil'] = np.ceil    # better for handling parse errors
BUILTIN_FUNCTIONS['frac'] = lambda x: x % 1.0


BUILTIN_CONSTANTS = {
    # Supported constants from MAD-X manual
    'pi': math.pi,
    'twopi': 2 * math.pi,
    'degrad': 180 / math.pi,  # °/rad
    'raddeg': math.pi / 180,  # rad/°
    'e': math.e,
    'emass': sc.electron_mass * sc.c**2 / sc.e,  # eV
    'pmass': sc.proton_mass * sc.c**2 / sc.e,  # eV
    'nmass': sc.neutron_mass * sc.c**2 / sc.e,  # eV
    'umass': sc.m_u * sc.c**2 / sc.e,  # eV
    'mumass': sc.value('muon mass') * sc.c**2 / sc.e,  # eV
    'clight': sc.c,  # m/s
    'qelect': sc.value('elementary charge'),  # A * s
    'hbar': sc.hbar,  # eV * s
    'erad': sc.value('classical electron radius'),  # m
    'prad': sc.value('classical electron radius') * (sc.m_e / sc.m_p),  # m
}

AVAILABLE_ELEMENT_CLASSES = {cls.__name__: cls for cls in xt_element_classes}

try:
    from xfields import element_classes as xf_element_classes
    AVAILABLE_ELEMENT_CLASSES.update({cls.__name__: cls for cls in xf_element_classes})
except ModuleNotFoundError:
    pass

try:
    from xcoll import element_classes as xc_element_classes
    AVAILABLE_ELEMENT_CLASSES.update({cls.__name__: cls for cls in xc_element_classes})
except ModuleNotFoundError:
    pass


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


class XMadParseError(Exception):
    LIMIT = 15

    def __init__(self, parse_log):
        self.parse_log = parse_log.copy()
        message = '\n\n'.join([repr(entry) for entry in parse_log[:self.LIMIT]])

        if len(parse_log) > self.LIMIT:
            message += f'\n\nTruncated {len(parse_log) - self.LIMIT} more errors...'

        super().__init__(message)


@cy.cclass
class Parser:
    scanner = cy.declare(xmad.yyscan_t)
    log = cy.declare(list, visibility='public')
    xd_manager = cy.declare(object, visibility='public')
    vars = cy.declare(dict, visibility='public')
    var_refs = cy.declare(object, visibility='public')
    elements = cy.declare(dict, visibility='public')
    element_refs = cy.declare(object, visibility='public')
    lines = cy.declare(dict, visibility='public')

    def __init__(self):
        self.log = []

        self.xd_manager = xd.Manager()

        self.vars = {}
        self.var_refs = self.xd_manager.ref(self.vars, 'vars')

        self.elements = {}
        self.element_refs = self.xd_manager.ref(self.elements, 'elements')

        self.lines = {}

        # So we do something interesting here: the objective is that we want to
        # keep a pointer to the parser object on the C side. However, for
        # reasons yet unknown, in the process of passing the pointer to C and
        # then taking it back, python decreases the reference count leading to
        # the premature death of the object (TODO: figure out why it happens?).
        #
        # It seems that we have two options: manually increasing the reference
        # count here, to inform Python that the object is in C, or to somehow
        # make this reference weak. The first solution is not ideal, as we
        # create a cycle: both the python-side parser and the C-side scanner
        # effectively hold strong references to one another.
        #
        # The implementation of the second solution is not super pretty, but it
        # is legitimate -- we cast to PyObject* to skip reference counting:
        xmad.yylex_init_extra(
            cy.cast(cy.pointer(PyObject), self),
            cy.address(self.scanner),
        )

    def __del__(self):
        xmad.yylex_destroy(self.scanner)

    def parse_string(self, string):
        xmad.yy_scan_string(string.encode(), self.scanner)
        success = xmad.yyparse(self.scanner)

        if success != 0 or self.log:
            raise XMadParseError(self.log)

    def parse_file(self, path):
        c_path = path.encode()

        file_readable = access(c_path, R_OK)
        if file_readable != 0:
            raise OSError(f'File `{path}` does not exist or is not readable.')

        xmad.yyset_in(fopen(c_path, "r"), self.scanner)

        success = xmad.yyparse(self.scanner)

        if success != 0 or self.log:
            raise XMadParseError(self.log)

    def handle_error(self, message):
        text = xmad.yyget_text(self.scanner).decode()
        yylloc_ptr = xmad.yyget_lloc(self.scanner)
        self.log.append(ParseLogEntry(yylloc_ptr[0], message, context=text))

    def get_identifier_ref(self, identifier):
        try:
            identifier = identifier.decode()
            if identifier not in self.vars:
                self.handle_error(f'use of an undefined variable `{identifier}`')
                return np.nan

            return self.var_refs[identifier]
        except Exception as e:
            print(f'######## identifier = {identifier} type = {type(identifier)}')
            traceback.print_exception(e)
            return np.nan

    def set_defer(self, identifier, value):
        self.var_refs[identifier] = value

    def set_value(self, identifier, value):
        if xd.refs.is_ref(value):
            value = value._get_value()

        self.var_refs[identifier] = value

    def add_line(self, line_name, elements, params):
        try:
            if elements is None or params is None:
                # A parsing error occurred, and we're in recovery.
                # Let's give up on making the line, it won't be any good anyway.
                return

            element_dict = {}
            local_element_refs = self.xd_manager.ref(element_dict, "element_refs")
            element_names = []
            for el_template in elements:
                name, parent, args = el_template
                element_cls = AVAILABLE_ELEMENT_CLASSES[parent]
                kwargs = {k: self._ref_get_value(v) for k, v in args}
                element = element_cls(**kwargs)
                element_dict[name] = element
                element_names.append(name)

                element_ref = local_element_refs[name]
                for k, v in args:
                    if isinstance(v, list):
                        list_ref = getattr(element_ref, k)
                        for i, le in enumerate(v):
                            if xd.refs.is_ref(le):
                                list_ref[i] = le
                    elif xd.refs.is_ref(v):
                        setattr(element_ref, k, v)

            line = xt.Line(elements=element_dict, element_names=element_names)
            line._var_management['lref'] = local_element_refs
            self.lines[line_name] = line
        except Exception as e:
            traceback.print_exception(e)

    @staticmethod
    def _ref_get_value(value_or_ref):
        if isinstance(value_or_ref, list):
            return [getattr(elem, '_value', elem) for elem in value_or_ref]
        return getattr(value_or_ref, '_value', value_or_ref)


def parser_from_scanner(yyscanner) -> Parser:
    # Cast back to Python, see comment in Parser.__init__:
    parser = xmad.yyget_extra(yyscanner)
    return cy.cast(Parser, parser)


def yyerror(_, yyscanner, message):
    parser = parser_from_scanner(yyscanner)
    parser.handle_error(message.decode())


@cy.exceptval(check=False)
def py_float(value):
    try:
        if KEEP_LITERAL_EXPRESSIONS:
            return LiteralExpr(value)
        return value
    except Exception as e:
        traceback.print_exception(e)


@cy.exceptval(check=False)
def py_unary_op(op_string, value):
    try:
        function = getattr(operator, op_string.decode())
        return function(value)
    except Exception as e:
        traceback.print_exception(e)


@cy.exceptval(check=False)
def py_binary_op(op_string, left, right):
    try:
        function = getattr(operator, op_string.decode())
        return function(left, right)
    except Exception as e:
        traceback.print_exception(e)


@cy.exceptval(check=False)
def py_call_func(scanner, func_name, value):
    try:
        normalized_name = func_name.decode().lower()
        if normalized_name not in BUILTIN_FUNCTIONS:
            parser_from_scanner(scanner).handle_error(
                f'builtin function `{normalized_name}` is unknown',
            )
            return np.nan

        function = BUILTIN_FUNCTIONS[normalized_name]
        return CallRef(function, (value,), {})
    except Exception as e:
        traceback.print_exception(e)


def py_eq_value_scalar(identifier, value):
    try:
        return identifier.decode(), value
    except Exception as e:
        traceback.print_exception(e)


def py_eq_defer_scalar(identifier, value):
    try:
        return identifier.decode(), value
    except Exception as e:
        traceback.print_exception(e)


def py_arrow(scanner, source_name, field_name):
    try:
        parser_from_scanner(scanner).handle_error(
            'the arrow syntax is not yet implemented'
        )
    except Exception as e:
        traceback.print_exception(e)


def py_identifier_atom(scanner, name):
    try:
        normalized_name = name.decode().lower()
        if name not in BUILTIN_CONSTANTS:
            parser = parser_from_scanner(scanner)
            return parser.get_identifier_ref(name)

        value = BUILTIN_CONSTANTS[normalized_name]
        return py_float(value)
    except Exception as e:
        traceback.print_exception(e)


def py_set_defer(scanner, assignment):
    try:
        parser = parser_from_scanner(scanner)
        parser.set_defer(*assignment)
    except Exception as e:
        traceback.print_exception(e)


@cy.exceptval(check=False)
def py_set_value(scanner, assignment):
    try:
        parser = parser_from_scanner(scanner)
        parser.set_value(*assignment)
    except Exception as e:
        traceback.print_exception(e)


def py_make_sequence(scanner, name, args, elements):
    try:
        parser = parser_from_scanner(scanner)
        parser.add_line(name.decode(), elements, dict(args))
    except Exception as e:
        traceback.print_exception(e)


def py_clone(name, parent, args) -> Tuple[str, str, dict]:
    try:
        return name.decode(), parent.decode(), args
    except Exception as e:
        traceback.print_exception(e)


def py_eq_value_sum(name, value) -> Tuple[str, object]:
    try:
        if xd.refs.is_ref(value):
            return name.decode(), value._get_value()
        return name.decode(), value
    except Exception as e:
        traceback.print_exception(e)


def py_eq_defer_sum(name, value) -> Tuple[str, object]:
    try:
        return name.decode(), value
    except Exception as e:
        traceback.print_exception(e)


def py_eq_value_array(name, array):
    try:
        return name.decode(), array
    except Exception as e:
        traceback.print_exception(e)


def py_eq_defer_array(name, array):
    try:
        return name.decode(), array
    except Exception as e:
        traceback.print_exception(e)
