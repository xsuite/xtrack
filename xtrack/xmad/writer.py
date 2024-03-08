import sys
import numpy as np
from typing import Any, Dict

import xdeps as xd
from xdeps.refs import XMadFormatter
import xtrack as xt


class XMadWriter:
    def __init__(self, name: str, line: xt.Line):
        self.name = name
        self.line = line
        self.var_manager: xd.Manager = line._var_management['manager']

    def write(self, stream=sys.stdout):
        previous_formatter = self.var_manager.formatter
        self.var_manager.formatter = XMadFormatter

        for var_name in self.line.vars.keys():
            if var_name.startswith('__'):
                continue

            value = self.line.vars[var_name]
            if value._expr:
                stream.write(f'{var_name:16} := {value._expr!r};\n')
            else:
                value = value._value
                if np.isscalar(value):
                    stream.write(f'{var_name:17} = {value};\n')
                else:
                    raise ValueError(
                        f'Cannot write the variable `{var_name}`, as values of '
                        f'type `{type(var_name)}` are not supported in XMad.')

        stream.write('\n')

        stream.write(f'{self.name}: sequence;\n')
        for element_name, element in self.line.items():
            element_dict = element.to_dict()
            element_type = element_dict.pop('__class__')

            stream.write(f'    {element_name}: {element_type}')

            element_ref = self.line.element_refs[element_name]
            expressions = self.var_manager.structure[element_ref]
            for ref in expressions:
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
