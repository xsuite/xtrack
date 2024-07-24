import builtins
import math
import operator
import re
import traceback
from collections import defaultdict
from typing import Tuple, Optional

import cython as cy
import numpy as np
import scipy.constants as sc
from cython.cimports.cpython.ref import Py_INCREF, PyObject  # noqa
from cython.cimports.libc.stdint import uintptr_t  # noqa
from cython.cimports.libc.stdio import fopen  # noqa
from cython.cimports.posix.unistd import access, R_OK  # noqa
from cython.cimports.xtrack.sequence import parser as xld  # noqa

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
            out += '\n    > ' + self.context
        return out


class XMadParseError(Exception):
    LIMIT = 15

    def __init__(self, parse_log):
        self.parse_log = parse_log.copy()

        message = 'Errors occurred while parsing:\n\n' if len(parse_log) > 1 else ''

        message += '\n\n'.join([repr(entry) for entry in parse_log[:self.LIMIT]])

        if len(parse_log) > self.LIMIT:
            message += f'\n\nTruncated {len(parse_log) - self.LIMIT} more errors...'

        super().__init__(message)


@cy.cfunc
def register_error(scanner: xld.yyscan_t, exception, action, add_context=True):
    parser = parser_from_scanner(scanner)
    caught_exc_string = '\n'.join(traceback.format_exception(exception))
    caught_exc_string = _indent_string(caught_exc_string, indent='    > ')

    full_error_string = (
        f"While {action} the following error occurred:\n\n"
        f"{caught_exc_string}"
    )
    parser.handle_error(full_error_string, add_context=add_context)


def _indent_string(string, indent='    '):
    return re.sub('^', indent, string, flags=re.MULTILINE)


@cy.cclass
class Parser:
    scanner = cy.declare(xld.yyscan_t)
    log = cy.declare(list, visibility='public')
    xd_manager = cy.declare(object, visibility='public')
    vars = cy.declare(object, visibility='public')
    var_refs = cy.declare(object, visibility='public')
    elements = cy.declare(dict, visibility='public')
    element_refs = cy.declare(object, visibility='public')
    lines = cy.declare(dict, visibility='public')
    global_elements = cy.declare(dict, visibility='public')
    _context = cy.declare(object, visibility='public')

    def __init__(self, _context):
        self.log = []

        self.xd_manager = xd.Manager()

        self.vars = defaultdict(lambda: 0)
        self.vars.default_factory = None
        self.var_refs = self.xd_manager.ref(self.vars, 'vars')

        self.elements = {}
        self.element_refs = self.xd_manager.ref(self.elements, 'eref')

        self.lines = {}
        self.global_elements = {}

        self._context = _context
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
        xld.yylex_init_extra(
            cy.cast(cy.pointer(PyObject), self),
            cy.address(self.scanner),
        )

    def __del__(self):
        xld.yylex_destroy(self.scanner)

    def parse_string(self, string):
        xld.yy_scan_string(string.encode(), self.scanner)

        # yy_scan_string doesn't reset the line and column, so we do it manually
        xld.yyset_lineno(1, self.scanner)
        xld.yyset_column(1, self.scanner)

        success = xld.yyparse(self.scanner)

        if success != 0 or self.log:
            raise XMadParseError(self.log)

    def parse_file(self, path):
        c_path = path.encode()

        file_readable = access(c_path, R_OK)
        if file_readable != 0:
            raise OSError(f'File `{path}` does not exist or is not readable.')

        xld.yyset_in(fopen(c_path, "r"), self.scanner)

        success = xld.yyparse(self.scanner)

        if success != 0 or self.log:
            raise XMadParseError(self.log)

    def get_line(self, name):
        if not self.lines:
            raise ValueError('No sequence was parsed. Either the input has no '
                             'sequences, or the parser has not been run yet.')

        if name is None:
            try:
                name, = self.lines.keys()
            except ValueError:
                raise ValueError('Cannot unambiguously determine the sequence '
                                 'as no name was provided and there is more '
                                 'than one sequence in parsed input.')

        return self.lines[name]

    def handle_error(self, message, add_context=True):
        text = xld.yyget_text(self.scanner).decode()
        yylloc_ptr = xld.yyget_lloc(self.scanner)
        log_entry = ParseLogEntry(yylloc_ptr[0], message)
        if add_context:
            log_entry.context = text
        self.log.append(log_entry)

    def get_identifier_ref(self, identifier):
        try:
            identifier = identifier.decode()
            if identifier not in self.vars:
                self.handle_error(f'use of an undefined variable `{identifier}`')
                return np.nan

            return self.var_refs[identifier]
        except Exception as e:
            traceback.print_exception(e)
            return np.nan

    def set_value(self, identifier, value):
        self.var_refs[identifier] = value

    def add_global_element(self, name, parent, args):
        self.global_elements[name] = (parent, args)

    def add_line(self, line_name, elements, params):
        try:
            if elements is None or params is None:
                # A parsing error occurred, and we're in recovery.
                # Let's give up on making the line, it won't be any good anyway.
                return

            self.elements[line_name] = local_elements = {}
            element_names = []

            for el_template in elements:
                if el_template is None:
                    # A parsing error occurred and has been registered already.
                    # Ignore this element.
                    continue

                name, parent, args = el_template

                self.add_element_to_line(line_name, name, parent, args)

                parent_name = getattr(local_elements[name], 'parent_name', None)
                if parent_name and parent_name not in local_elements:
                    self.add_element_to_line(
                        line_name,
                        parent_name,
                        *self.global_elements[parent_name],
                    )

                element_names.append(name)

            line = xt.Line(
                elements=self.elements[line_name],
                element_names=element_names,
            )
            line._var_management['vref'] = self.var_refs
            line._var_management['lref'] = self.element_refs[line_name]
            self.lines[line_name] = line
        except Exception as e:
            register_error(self.scanner, e, 'building the sequence')

    def add_element_to_line(self, line_name, name, parent, args):
        local_elements = self.elements[line_name]

        if not args and parent not in AVAILABLE_ELEMENT_CLASSES:  # simply insert a replica
            local_elements[name] = xt.Replica(parent_name=parent)
            return

        element_cls = AVAILABLE_ELEMENT_CLASSES[parent]
        kwargs = {k: self._ref_get_value(v) for k, v in args}
        element = element_cls.from_dict(kwargs, _context=self._context)
        local_elements[name] = element
        element_ref = self.element_refs[line_name][name]

        for k, v in args:
            if isinstance(v, list):
                list_ref = getattr(element_ref, k)
                for i, le in enumerate(v):
                    if xd.refs.is_ref(le):
                        list_ref[i] = le
            elif xd.refs.is_ref(v):
                setattr(element_ref, k, v)

    @staticmethod
    def _ref_get_value(value_or_ref):
        if isinstance(value_or_ref, list):
            return [getattr(elem, '_value', elem) for elem in value_or_ref]
        return getattr(value_or_ref, '_value', value_or_ref)


def parser_from_scanner(yyscanner) -> Parser:
    # Cast back to Python, see comment in Parser.__init__:
    parser = xld.yyget_extra(yyscanner)
    return cy.cast(Parser, parser)


def yyerror(_, yyscanner, message):
    parser = parser_from_scanner(yyscanner)
    parser.handle_error(message.decode())


def py_float(scanner, value):
    return py_numeric(scanner, value)


def py_integer(scanner, value):
    return py_numeric(scanner, value)


@cy.exceptval(check=False)
def py_numeric(scanner, value):
    try:
        if KEEP_LITERAL_EXPRESSIONS:
            return LiteralExpr(value)
        return value
    except Exception as e:
        register_error(scanner, e, f'parsing a numeric value')


@cy.exceptval(check=False)
def py_unary_op(scanner, op_string, value):
    try:
        function = getattr(operator, op_string.decode())
        return function(value)
    except Exception as e:
        register_error(scanner, e, f'parsing a unary operation')


@cy.exceptval(check=False)
def py_binary_op(scanner, op_string, left, right):
    try:
        function = getattr(operator, op_string.decode())
        return function(left, right)
    except Exception as e:
        register_error(scanner, e, f'parsing a binary operation')


@cy.exceptval(check=False)
def py_call_func(scanner, func_name, value):
    try:
        normalized_name = func_name.decode()
        if normalized_name == 'const':
            return value._value if xd.refs.is_ref(value) else value
        elif normalized_name not in BUILTIN_FUNCTIONS:
            parser_from_scanner(scanner).handle_error(
                f'builtin function `{normalized_name}` is unknown',
            )
            return np.nan

        function = BUILTIN_FUNCTIONS[normalized_name]
        return CallRef(function, (value,), {})
    except Exception as e:
        register_error(scanner, e, f'parsing a function call')


def py_eq_value_scalar(scanner, identifier, value):
    try:
        return identifier.decode(), value
    except Exception as e:
        register_error(scanner, e, f'parsing a deferred scalar assignment')


def py_arrow(scanner, source_name, field_name):
    try:
        raise NotImplementedError
    except Exception as e:
        register_error(scanner, e, f'parsing the arrow syntax')


def py_identifier_atom(scanner, name):
    try:
        normalized_name = name.decode()
        if normalized_name not in BUILTIN_CONSTANTS:
            parser = parser_from_scanner(scanner)
            return parser.get_identifier_ref(name)

        value = BUILTIN_CONSTANTS[normalized_name]
        return py_float(scanner, value)
    except Exception as e:
        register_error(scanner, e, f'parsing an identifier')


def py_set_value(scanner, assignment):
    try:
        parser = parser_from_scanner(scanner)
        parser.set_value(*assignment)
    except Exception as e:
        register_error(scanner, e, 'parsing a deferred assign statement')


def py_make_sequence(scanner, name, args, elements):
    try:
        parser = parser_from_scanner(scanner)
        parser.add_line(name.decode(), elements, dict(args))
    except Exception as e:
        register_error(scanner, e, 'parsing a sequence')


def py_clone(scanner, name, parent, args) -> Optional[Tuple[str, str, dict]]:
    try:
        if name.decode() in AVAILABLE_ELEMENT_CLASSES:
            parser = parser_from_scanner(scanner)
            parser.handle_error(f'the name `{name.decode()}` shadows a built-in type.')
            return None

        return name.decode(), parent.decode(), args
    except Exception as e:
        register_error(scanner, e, 'parsing a clone statement')


def py_eq_value_sum(scanner, name, value) -> Tuple[str, object]:
    try:
        return name.decode(), value
    except Exception as e:
        register_error(scanner, e, 'parsing a deferred sum statement')


def py_eq_value_array(scanner, name, array):
    try:
        return name.decode(), array
    except Exception as e:
        register_error(scanner, e, 'parsing a deferred array assignment')


def py_clone_global(scanner, clone):
    try:
        if clone is None:  # A parsing error already occurred, recover
            return
        parser = parser_from_scanner(scanner)
        name, parent, args = clone
        parser.add_global_element(name, parent, args)
    except Exception as e:
        register_error(scanner, e, 'parsing a global clone statement')