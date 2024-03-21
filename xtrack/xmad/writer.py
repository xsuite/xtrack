import sys
from collections import defaultdict
from functools import reduce

import numpy as np
from typing import Any, Dict, Callable

import xdeps as xd
from xdeps.refs import XMadFormatter
import xtrack as xt
from xdeps.sorting import toposort, reverse_graph


class XMadWriter:
    def __init__(self, name: str, line: xt.Line):
        self.name = name
        self.line = line
        self.var_manager: xd.Manager = line._var_management['manager']

    def write(self, stream=sys.stdout):
        previous_formatter = self.var_manager.formatter
        self.var_manager.formatter = XMadFormatter

        for var_name in self._sorted_vars():
            if var_name.startswith('__'):
                continue

            value = self.line.vars[var_name]

            if value._expr:
                stream.write(f'{var_name:16} := {value._expr!r};\n')
            else:
                value = value._value
                if np.isscalar(value):
                    stream.write(f'{var_name:17} = {value};\n')
                elif isinstance(value, Callable):
                    continue
                else:
                    raise ValueError(
                        f'Cannot write the variable `{var_name}`, as values of '
                        f'type `{type(value).__name__}` (here {value}) are '
                        f'not supported in XMad.')

        stream.write('\n')

        stream.write(f'{self.name}: sequence;\n')
        for element_name, element in self.line.items():
            # TODO: We should somehow import elements in a better way from
            #   MAD-X to avoid this:
            element_name = element_name.replace(':', '_')

            element_dict = element.to_dict()
            element_type = element_dict.pop('__class__')

            stream.write(f'    {element_name}: {element_type}')

            element_ref = self.line.element_refs[element_name]
            expressions = self.var_manager.structure[element_ref]
            for ref in expressions:
                if ref._key not in element._xo_fnames:
                    # This is not great! Some elements can apparently be imported
                    # as drifts (solenoid), but their expressions (ks) are still
                    # attached, making the definition of the element invalid.
                    # TODO: We should fix this.
                    continue
                element_dict[ref._key] = ref._expr

            if element_dict:
                args = self._format_arglist(element_dict)
                stream.write(', ' + args)
            stream.write(';\n')

        stream.write(f'endsequence;\n')

        self.var_manager.formatter = previous_formatter

    def _format_arglist(self, args: Dict[str, Any]) -> str:
        formatted_args = []
        for arg_name, arg_value in args.items():
            if arg_value is True:
                formatted_args.append(arg_name)
            elif arg_value is False:
                formatted_args.append(f'-{arg_name}')
            elif xd.refs.is_ref(arg_value):
                formatted_args.append(f'{arg_name} := {arg_value!r}')
            elif isinstance(arg_value, str):
                escaped_value = arg_value.replace('"', '\\"')
                formatted_args.append(f'{arg_name} = "{escaped_value}"')
            elif np.isscalar(arg_value):
                arg_value = self.scalar_to_str(arg_value)
                formatted_args.append(f'{arg_name} = {arg_value}')
            else:  # is some kind of list
                formatted_args.append(f'{arg_name} = ' + '{' + self._build_list(arg_value) + '}')

        if len(formatted_args) <= 1:
            return ', '.join(formatted_args)
        else:
            return '\n        ' + ',\n        '.join(formatted_args)

    def _build_list(self, list_like) -> str:
        return ', '.join(map(self.scalar_to_str, list_like))

    @staticmethod
    def scalar_to_str(scalar):
        with np.printoptions(floatmode='unique'):
            return str(scalar)

    def _sorted_vars(self):
        var_graph = {}
        for name in self.line.vars.keys():
            ref = self.line.vars[name]
            expr = self.line.vars[name]._expr
            if expr is not None:
                var_graph[ref] = expr._get_dependencies()
            else:
                var_graph[ref] = set()

        initial = reduce(set.union, var_graph.values(), set(var_graph.keys()))

        sorted_vars = toposort(var_graph, initial)
        sorted_keys = [var._key for var in sorted_vars]

        return sorted(self.line.vars.keys(), key=lambda k: -sorted_keys.index(k))
