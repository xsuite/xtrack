import json
import math
import operator
import re
import traceback
from collections import defaultdict
from pathlib import Path
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
import xobjects as xo
import xtrack as xt
from xdeps.refs import CallRef, LiteralExpr, XldFormatter
from xtrack.beam_elements import element_classes as xt_element_classes
import xtrack.sequence.string_formatting as fmt

KEEP_LITERAL_EXPRESSIONS = False


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


class InvalidExpression:
    """A placeholder class for when an expression is expected but no valid
    expression can be given."""
    @classmethod
    def _set_to_expr(cls, _):
        """A bad expression error was reported by this point; do nothing."""
        pass


@cy.cfunc
def _ref_get_value(value_or_ref):
    if isinstance(value_or_ref, list):
        return [getattr(elem, '_value', elem) for elem in value_or_ref]
    return getattr(value_or_ref, '_value', value_or_ref)


class ParseLogEntry:
    def __init__(self, location, reason, error=True, context=None):
        self.line_no = location['first_line']
        self.column = location['first_column']
        self.end_line_no = location['last_line']
        self.end_column = location['last_column']
        self.reason = reason
        self.context = context
        self.error = error

    def add_file_line(self, file_lines):
        relevant_lines = file_lines[self.line_no - 1:self.end_line_no]

        def _insert(string, where, what):
            return string[:where] + what + string[where:]

        relevant_lines[-1] = _insert(relevant_lines[-1], self.end_column - 1, fmt.NO_UNDERLINE)
        relevant_lines[0] = _insert(relevant_lines[0], self.column - 1, fmt.UNDERLINE)

        self.context = '\n'.join(relevant_lines)

    def __repr__(self):
        if self.error:
            out = fmt.error('Error')
        else:
            out = fmt.warning('Warning')
        out += f' on line {self.line_no} column {self.column}: {self.reason}'

        if self.context:
            out += '\n\n'
            out += fmt.indent_string(self.context, indent='    > ')

        return out


class ParseError(Exception):
    LIMIT = 15

    def __init__(self, parse_log, error_type="error", file_lines=None):
        parse_log = parse_log.copy()

        if len(parse_log) > 1:
            message = f'{error_type.title()}s occurred while parsing:\n\n'
        else:
            message = ''

        truncated = parse_log[:self.LIMIT]
        for entry in truncated:
            entry.add_file_line(file_lines)
        message += '\n\n'.join([repr(entry) for entry in truncated])

        if len(parse_log) > self.LIMIT:
            message += (
                f'\n\nTruncated {len(parse_log) - self.LIMIT} more '
                f'{error_type}s...'
            )

        super().__init__(message)


@cy.cfunc
def register_error(scanner: xld.yyscan_t, exception, action, add_context=True, location=None):
    parser = parser_from_scanner(scanner)
    caught_exc_string = '\n'.join(traceback.format_exception(exception))
    caught_exc_string = fmt.indent_string(caught_exc_string, indent='    > ')

    full_error_string = (
        f"While {action} the following error occurred:\n\n"
        f"{caught_exc_string}"
    )
    parser.handle_error(full_error_string, add_context=add_context, location=location)


@cy.cclass
class LineTemplate:
    name = cy.declare(str, visibility='public')
    args = cy.declare(list, visibility='public')
    element_dict = cy.declare(dict, visibility='public')
    element_names = cy.declare(list, visibility='public')
    line_attributes = cy.declare(dict, visibility='public')
    location = cy.declare(object, visibility='public')
    parser = cy.declare(object, visibility='public')

    def __init__(self, name, args, location):
        self.name = name
        self.args = args
        self.element_dict = {}
        self.element_names = []
        self.line_attributes = {}
        self.location = location

    def add_element(self, name, parent, args):
        self._add_element(name, parent, args)

        parent_name = getattr(self.element_dict[name], 'parent_name', None)
        if parent_name and parent_name not in self.element_dict:
            self._add_element(
                parent_name,
                *self.parser.global_elements[parent_name],
            )

        self.element_names.append(name)

    def set_parser(self, parser):
        self.parser = parser

        # Expressions need to go through the parser to be evaluated, so we
        # need to update the element references in the parser with ours.
        parser.elements[self.name] = self.element_dict

    def _add_element(self, name, parent, args):
        if not args and parent not in AVAILABLE_ELEMENT_CLASSES:  # simply insert a replica
            self.element_dict[name] = xt.Replica(parent_name=parent)
            return

        if parent not in AVAILABLE_ELEMENT_CLASSES:
            if parent in self.parser.global_elements:
                raise SyntaxError(
                    f'Cloning elements while overriding parameters is not yet '
                    f'supported in the line.'
                )
            raise SyntaxError(f'Unknown element class `{parent}`.')

        element_cls = AVAILABLE_ELEMENT_CLASSES[parent]
        kwargs = {k: _ref_get_value(v) for k, v in args}
        element = element_cls.from_dict(kwargs, _context=self.parser._context)
        self.element_dict[name] = element  # this enables the leak somehow
        element_ref = self.parser.element_refs[self.name][name]

        for k, v in args:
            if isinstance(v, list):
                list_ref = getattr(element_ref, k)
                for i, le in enumerate(v):
                    if xd.refs.is_ref(le):
                        list_ref[i] = le
            elif xd.refs.is_ref(v):
                setattr(element_ref, k, v)

    def add_command(self, command, args):
        argdict = {k: _ref_get_value(v) for k, v in args}
        if command == 'particle_ref':
            return self.add_particle_ref(**argdict)
        if command == 'twiss_default':
            return self.add_twiss_default(**argdict)
        if command == 'config':
            return self.add_config(**argdict)
        if command == 'attr':
            return self.add_attribute(**argdict)
        raise TypeError(f'Unknown command `{command}`.')

    def add_particle_ref(self, **kwargs):
        particle_ref = xt.Particles(**kwargs)
        self.line_attributes['particle_ref'] = particle_ref

    def add_twiss_default(self, **kwargs):
        json_part = json.loads(kwargs.pop('json', "{}"))
        kwargs.update(json_part)
        self.line_attributes['twiss_default'] = kwargs

    def add_config(self, **kwargs):
        self.line_attributes['config'] = kwargs

    def add_attribute(self, **kwargs):
        if kwargs.keys() != {'update', 'json'}:
            raise ValueError('The `attr` command expects `update` and `json` parameters.')
        self.line_attributes[kwargs['update']] = json.loads(kwargs['json'])


@cy.cclass
class Parser:
    scanner = cy.declare(xld.yyscan_t)
    log = cy.declare(list, visibility='public')
    xd_manager = cy.declare(object, visibility='public')
    vars = cy.declare(object, visibility='public')
    var_refs = cy.declare(object, visibility='public')
    elements = cy.declare(dict, visibility='public')
    element_refs = cy.declare(object, visibility='public')
    functions = cy.declare(object, visibility='public')
    func_refs = cy.declare(object, visibility='public')
    lines = cy.declare(dict, visibility='public')
    global_elements = cy.declare(dict, visibility='public')

    _current_line_template = cy.declare(object, visibility='public')
    _context = cy.declare(object, visibility='public')
    _single_line_mode: bool

    def __init__(self, single_line_mode=False, _context=xo.context_default):
        self.log = []

        self.xd_manager = xd.Manager()

        # This is apparently expected in xt.Line, so we provide it instead of {}
        self.vars = defaultdict(lambda: None)

        self.var_refs = self.xd_manager.ref(self.vars, 'vars')

        self.elements = {}
        self.element_refs = self.xd_manager.ref(self.elements, 'element_refs')

        self.functions = xt.line.Functions()
        self.func_refs = self.xd_manager.ref(self.functions, 'f')

        self.lines = {}
        self.global_elements = {}

        self._current_line_template = None
        self._context = _context
        self._single_line_mode = single_line_mode
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

        self._assert_success(success, string)

    def parse_file(self, path):
        c_path = path.encode()

        file_readable = access(c_path, R_OK)
        if file_readable != 0:
            raise OSError(f'File `{path}` does not exist or is not readable.')

        xld.yyset_in(fopen(c_path, "r"), self.scanner)

        success = xld.yyparse(self.scanner)

        self._assert_success(success, Path(path))

    def _assert_success(self, success, input_file_or_string=None):
        errors = [entry for entry in self.log if entry.error]

        def _make_error(error_type='error'):
            input = input_file_or_string
            if isinstance(input_file_or_string, Path):
                input = input_file_or_string.read_text()
            parse_error = ParseError(
                self.log, error_type=error_type, file_lines=input.split('\n'))
            return parse_error

        if success != 0 or errors:
            raise _make_error()

        if self.log:
            parse_warning = _make_error(error_type='warning')
            xt.general._print(str(parse_warning))

        self.log = []  # reset log after all errors have been reported

    def handle_error(self, message, add_context=True, location=None):
        self._handle_parse_log_event(
            message, is_error=True, add_context=add_context, location=location)

    def handle_warning(self, message, add_context=True, location=None):
        self._handle_parse_log_event(
            message, is_error=False, add_context=add_context, location=location)

    def _handle_parse_log_event(self, message, is_error, add_context, location):
        text = xld.yyget_text(self.scanner).decode()
        yylloc_ptr = xld.yyget_lloc(self.scanner)
        location = location or yylloc_ptr[0]
        log_entry = ParseLogEntry(location, message, error=is_error)
        if add_context:
            log_entry.context = text
        self.log.append(log_entry)

    def get_identifier_ref(self, identifier, location):
        try:
            if identifier in self.vars:
                return self.var_refs[identifier]

            if identifier in BUILTIN_CONSTANTS:
                return py_float(self.scanner, BUILTIN_CONSTANTS[identifier])

            self.handle_error(
                f'use of an undefined variable `{identifier}`',
                location=location,
            )
            return np.nan
        except Exception as e:
            register_error(self.scanner, e, 'getting an identifier reference')

    def set_value(self, identifier, value, location):
        if identifier in self.vars:
            self.handle_warning(
                f'redefinition of the variable `{identifier}`',
                add_context=True,
                location=location,
            )
        self.var_refs[identifier] = value

    def add_global_element(self, name, parent, args):
        self.global_elements[name] = (parent, args)

    def open_line(self, line_template):
        line_template.set_parser(self)
        self._current_line_template = line_template

    def commit_line(self, line_template):
        if line_template is not self._current_line_template:
            # I cannot think of a hypothetical scenario where this would happen
            self.handle_error('internal error occurred when building the line')
            return

        self._current_line_template = None

        line_name = line_template.name
        line = xt.Line(
            elements=line_template.element_dict,
            element_names=line_template.element_names,
        )
        line._var_management = {}
        line._var_management['data'] = {}
        line._var_management['data']['var_values'] = self.vars
        line._var_management['data']['functions'] = self.functions

        line._var_management['manager'] = self.xd_manager
        line._var_management['vref'] = self.var_refs
        line._var_management['lref'] = self.element_refs[line_name]
        line._var_management['fref'] = self.func_refs

        for attr_name, attr_value in line_template.line_attributes.items():
            if hasattr(getattr(line, attr_name, None), 'update'):
                getattr(line, attr_name).update(attr_value)
            else:
                setattr(line, attr_name, attr_value)

        self.lines[line_name] = line

    def get_reference(self, parent, field, location):
        """Return the result of the syntax: parent->field

        This is context dependent. If parent is None, assume that field names
        element refs of a line/sequence with such name. If parent is a dict
        or a list (element refs, a list attribute of an element) this is
        equivalent to parent[field], and otherwise to parent.field.

        Arguments
        ---------
        parent: xd.MutableRef
        field: str
        """
        if parent is InvalidExpression:
            return InvalidExpression

        def _error(message):
            self.handle_error(message, add_context=True, location=location)

        if parent is None:  # In this case assume
            # Currently we do not allow this sort of assignment within the
            # line/sequence, so this should not ever happen
            assert self._current_line_template is None

            if field not in self.elements and not self._single_line_mode:
                _error(f"no such line/sequence: `{field}`")
                return InvalidExpression

            return self.element_refs[field]

        try:
            parent_value = parent._value
        except (KeyError, AttributeError):
            formatter = XldFormatter(scope=self.element_refs['line'])
            _error(f"cannot access `{field}` as "
                   f"`{parent._formatted(formatter)}` does not exist")
            return InvalidExpression

        if isinstance(parent_value, dict):
            return parent[field]

        if isinstance(parent_value, list):
            try:
                return parent[int(field)]
            except ValueError:
                formatter = XldFormatter(scope=None)
                _error(f"`{parent._formatted(formatter)}` is a list, an "
                       f"integer was expected instead of `{field}`")

        return getattr(parent, field)


    def get_line(self, name, copy_manager=True):
        """Get a copy of the parsed sequence with the given name.

        If `copy_manager` is True, the new line will have its own variable
        management, so changes applied to its variables will not affect the
        other lines in the parser. This way, we obtain a proper xt.Line object.

        If `copy_manager` is False, the new line will share the variable
        manager (and in particular its element_refs will not be technically
        as expected of a regular xt.Line), which may cause unexpected behavior
        if the line is copied.

        If an ensemble of lines sharing the same variable manager is needed,
        consider obtaining an xt.Multiline object instead with `get_multiline`.
        """
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

        line = self.lines[name]
        if not copy_manager:
            return line

        line = line.copy(include_var_management=False)
        line._init_var_management()
        new_line_manager = line._xdeps_manager
        shared_manager = self.xd_manager

        line._xdeps_vref._owner.update(self.vars)

        new_line_manager.copy_expr_from(shared_manager, 'vars')
        new_line_manager.copy_expr_from(shared_manager, 'f')
        new_line_manager.copy_expr_from(
            shared_manager,
            'element_refs',
            bindings={self.element_refs[name]: line.element_refs},
        )

        return line

    def set_existing_line(self, line, name):
        if self.lines or not self._single_line_mode:
            raise ValueError(
                'An existing line can only be added to a fresh parser in '
                'single line mode. This parser was either created with '
                '`single_line_mode=False or `set_existing_line` method has '
                'been called before.'
            )
        self.xd_manager = line._xdeps_manager
        self.lines[name] = line
        self.elements[name] = line.element_dict
        self.element_refs = line.element_refs
        self.vars = line._var_management['data']['var_values']
        self.var_refs = line._var_management['vref']
        self.functions = line._var_management['data']['functions']
        self.func_refs = line._var_management['fref']

    def get_multiline(self):
        multiline = xt.Multiline(lines=self.lines, link_vars=False)
        multiline._var_sharing = xt.multiline.VarSharing(
            lines={},
            names=[],
            existing_manager=self.xd_manager,
            existing_vref=self.var_refs,
            existing_eref=self.element_refs,
            existing_fref=self.func_refs,
        )
        multiline._multiline_vars = xt.line.LineVars(multiline)
        for name, line in self.lines.items():
            line._var_management = None

        return multiline


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
def py_string(scanner, value):
    try:
        return re.sub(r'\\"', '"', value.decode())
    except Exception as e:
        register_error(scanner, e, f'parsing a string')


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
        name = func_name.decode()
        if name == 'const':
            return value._value if xd.refs.is_ref(value) else value

        parser = parser_from_scanner(scanner)
        if name not in parser.functions:
            parser_from_scanner(scanner).handle_error(
                f'builtin function `{name}` is unknown',
            )
            return np.nan

        return getattr(parser.func_refs, name)(value)
    except Exception as e:
        register_error(scanner, e, f'parsing a function call')


def py_assign(scanner, identifier, value):
    try:
        return identifier.decode(), value
    except Exception as e:
        register_error(scanner, e, f'parsing an assignment')


def py_arrow(scanner, source_name, field_name):
    parser = parser_from_scanner(scanner)
    parser.handle_error(
        'expressions with -> are not yet implemented'
    )


def py_identifier_atom(scanner, name, location):
    try:
        parser = parser_from_scanner(scanner)
        return parser.get_identifier_ref(name.decode(), location)
    except Exception as e:
        register_error(
            scanner, e, f'parsing an identifier',
            add_context=True, location=location,
        )


def py_set_value(scanner, assignment, location):
    try:
        identifier, value = assignment
        parser = parser_from_scanner(scanner)
        if identifier in BUILTIN_CONSTANTS:
            parser.handle_warning(
                f"variable `{identifier}` shadows a built-in constant",
                add_context=False,
                location=location,
            )
        parser.set_value(identifier, value, location)
    except Exception as e:
        register_error(
            scanner, e, 'parsing a deferred assign statement',
            add_context=True, location=location,
        )


def py_reference(scanner, parent, field, location):
    try:
        parser = parser_from_scanner(scanner)
        return parser.get_reference(parent, field.decode(), location)
    except Exception as e:
        register_error(scanner, e, 'accessing a reference field')


def py_set_ref(scanner, target, value):
    try:
        target._set_to_expr(value)
    except Exception as e:
        register_error(scanner, e, 'setting a field reference')


def py_start_beamline(scanner, name, args, location):
    try:
        line_template = LineTemplate(name.decode(), args, location)
        parser = parser_from_scanner(scanner)
        parser.open_line(line_template)
        return line_template
    except Exception as e:
        register_error(scanner, e, 'parsing a sequence header')


def py_end_beamline(scanner, line_template):
    try:
        parser = parser_from_scanner(scanner)
        parser.commit_line(line_template)
    except Exception as e:
        register_error(scanner, e, 'parsing a sequence')


def py_clone(scanner, name, command) -> Optional[Tuple[str, str, dict]]:
    try:
        if name.decode() in AVAILABLE_ELEMENT_CLASSES:
            parser = parser_from_scanner(scanner)
            parser.handle_error(f'the name `{name.decode()}` shadows a built-in type.')
            return None

        return name.decode(), command[0], command[1]
    except Exception as e:
        register_error(scanner, e, 'parsing a clone statement')


def py_add_element(scanner, target, element):
    try:
        target.add_element(*element)
    except Exception as e:
        register_error(scanner, e, 'building a new element')


def py_clone_global(scanner, clone):
    try:
        if clone is None:  # A parsing error already occurred, recover
            return
        parser = parser_from_scanner(scanner)
        name, parent, args = clone
        parser.add_global_element(name, parent, args)
    except Exception as e:
        register_error(scanner, e, 'parsing a global clone statement')


def py_command(scanner, name, arguments, location):
    try:

        return name.decode(), arguments
    except Exception as e:
        register_error(scanner, e, 'parsing a command statement',
                       add_context=True, location=location)


def py_add_command(scanner, target, command, location):
    try:
        target.add_command(*command)
    except Exception as e:
        register_error(scanner, e, 'applying a command statement',
                       add_context=True, location=location)
