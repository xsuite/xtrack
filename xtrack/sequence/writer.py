from xtrack.general import _print
import sys
import sys
from functools import reduce
from typing import Any, Dict, Callable, Union, Tuple, Set
from typing import Optional

import numpy as np

import xdeps as xd
import xtrack as xt
from xdeps.refs import XldFormatter, is_ref
from xdeps.sorting import toposort
from xtrack.general import _print
from xtrack.sequence.parser import BUILTIN_CONSTANTS

TAB_WIDTH = 4


class XMadWriter:
    def __init__(
            self,
            lattice: Union[xt.Line, xt.Multiline],
            name: Optional[str] = None
    ):
        if isinstance(lattice, xt.Line):
            self.lines = {name: lattice}
        elif isinstance(lattice, xt.Multiline):
            if name is not None:
                raise ValueError('Name is not needed for Multiline, as names of'
                                 'the lines are stored within it.')
            self.lines = lattice.lines
        else:
            raise ValueError(f'Expected a Line or Multiline, got {type(lattice)}.')

        self.vars = lattice.vars
        self.var_manager: xd.Manager = lattice._xdeps_manager

    def write(self, stream=sys.stdout):
        if self._write_expressions(stream):
            stream.write('\n')

        if self._write_templates(stream):
            stream.write('\n')

        for name, line in self.lines.items():
            self._write_line(stream, name, line)
            stream.write('\n')

    def _write_expressions(self, stream):
        formatter = XldFormatter(scope=None)

        wrote_stuff = False

        for var_name in self._sorted_vars():
            if var_name.startswith('__'):
                continue

            value = self.vars[var_name]
            if var_name in BUILTIN_CONSTANTS:
                lhs = BUILTIN_CONSTANTS[var_name]
                rhs = value._value
                if np.isclose(lhs, rhs, atol=1e-19):
                    continue
                _print(
                    f"Warning: The definition of `{var_name}` (= {rhs}) in "
                    f"your line shadows a built in constant of the same "
                    f"name (= {lhs}). Make sure this is desired!"
                )

            if value._expr:
                formatted_value = value._expr._formatted(formatter)
                stream.write(f'{var_name} = {formatted_value};\n')
                wrote_stuff = True
            else:
                value = value._value
                if np.isscalar(value):
                    stream.write(f'{var_name} = {value};\n')
                    wrote_stuff = True
                elif isinstance(value, Callable):
                    continue
                else:
                    raise ValueError(
                        f'Cannot write the variable `{var_name}`, as values of '
                        f'type `{type(value).__name__}` (here {value}) are '
                        f'not supported in XLD.')

        return wrote_stuff

    def _write_line(self, stream, name, line):
        formatter = XldFormatter(scope=line.element_refs)

        stream.write(f'{name}: sequence;\n')
        for element_name, element in line.items():
            self._write_element(stream, element_name, line, formatter, level=1)

        stream.write(f'endsequence;\n')

    def _write_templates(self, stream):
        formatter = XldFormatter(scope=None)
        templates_to_write: Set[Tuple[str, str]] = set()

        for line_name, line in self.lines.items():
            for element_name, element in line.items():
                if isinstance(element, xt.Replica) or hasattr(element, 'parent_name'):
                    templates_to_write.add((element.parent_name, line_name))

        for element_name, line_name in templates_to_write:
            line = self.lines[line_name]
            self._write_global_template(stream, element_name, line, formatter)

        return bool(templates_to_write)

    def _write_global_template(self, stream, element_name, line, formatter):
        self._write_element(stream, element_name, line, formatter, level=0)

    def _write_element(self, stream, element_name, line, formatter, level):
        # TODO: We should somehow import elements in a better way from
        #   MAD-X to avoid this:
        indent = ' ' * (level * TAB_WIDTH)
        element = line[element_name]
        element_name = element_name.replace(':', '_')
        element_dict = element.to_dict()

        if isinstance(element, xt.Replica):
            stream.write(f'{indent}{element_name}: {element.parent_name};')
            return

        element_type = element_dict.pop('__class__')

        stream.write(f'{indent}{element_name}: {element_type}')
        element_ref = line.element_refs[element_name]
        expressions = self.var_manager.structure[element_ref]

        for ref in expressions:
            if ref._key not in element._xo_fnames:
                # This is not great! Some elements can apparently be imported
                # as drifts (solenoid), but their expressions (ks) are still
                # attached, making the definition of the element invalid.
                # TODO: We should fix this.
                _print(f'Warning: {element_name} has no attribute {ref._key}, '
                       f'but an expression setting it is attached.')
                continue
            if ref._expr:
                element_dict[ref._key] = ref._expr
            else:  # it's probably an array
                element_dict[ref._key] = [
                    ref[ii]._expr or ref[ii]._value for ii in range(len(ref._value))
                ]

        if element_dict:
            args = self._format_arglist(element_dict, formatter, level=level + 1)
            stream.write(', ' + args)

        stream.write(';\n')

    def _format_arglist(self, args: Dict[str, Any], formatter: XldFormatter, level) -> str:
        formatted_args = []
        for arg_name, arg_value in args.items():
            if arg_value is True:
                formatted_args.append(arg_name)
            elif arg_value is False:
                formatted_args.append(f'-{arg_name}')
            elif is_ref(arg_value):
                formatted_value = arg_value._formatted(formatter)
                formatted_args.append(f'{arg_name} = {formatted_value}')
            elif isinstance(arg_value, str):
                escaped_value = arg_value.replace('"', '\\"')
                formatted_args.append(f'{arg_name} = "{escaped_value}"')
            elif np.isscalar(arg_value):
                arg_value = self.scalar_to_str(formatter)(arg_value)
                formatted_args.append(f'{arg_name} = {arg_value}')
            else:  # is some kind of list
                formatted_list = self._build_list(arg_value, formatter)
                formatted_args.append(f'{arg_name} = ' + '{' + formatted_list + '}')

        indent = ' ' * (level * TAB_WIDTH)
        if len(formatted_args) <= 1:
            return ', '.join(formatted_args)
        else:
            return f'\n{indent}' + f',\n{indent}'.join(formatted_args)

    def _build_list(self, list_like, formatter: XldFormatter) -> str:
        return ', '.join(map(self.scalar_to_str(formatter), list_like))

    @staticmethod
    def scalar_to_str(formatter: XldFormatter):
        def _mapper(scalar):
            with np.printoptions(floatmode='unique'):
                if is_ref(scalar):
                    return scalar._formatted(formatter)
                return str(scalar)
        return _mapper

    def _sorted_vars(self):
        var_graph = {}
        for name in self.vars.keys():
            ref = self.vars[name]
            expr = self.vars[name]._expr
            if expr is not None:
                var_graph[ref] = expr._get_dependencies()
            else:
                var_graph[ref] = set()

        initial = reduce(set.union, var_graph.values(), set(var_graph.keys()))

        sorted_vars = toposort(var_graph, initial)
        sorted_keys = [var._key for var in sorted_vars]

        return sorted(self.vars.keys(), key=lambda k: -sorted_keys.index(k))
