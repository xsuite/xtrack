# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #
from os import PathLike

import xtrack as xt


class _Code:
    """Wrapper for a string to print a repr without quotes."""
    def __init__(self, code):
        self.code = code

    def __repr__(self):
        return self.code


class EnvWriterProxy:
    """An object with an API similar to an Environment but it simply writes
    the operations performed on it."""

    def __init__(self, env=None, prefix: str = 'env', parent_proxy: 'EnvWriterProxy' = None):
        print('WARNING: EnvWriterProxy is an experimental feature, which may '
              'not work correctly, and change in the future.')
        self.env = env or xt.Environment()
        self.prefix = prefix

        if parent_proxy:
            self._temp_lines = parent_proxy._temp_lines
            self._lines = parent_proxy._lines
        else:
            self._temp_lines = {}
            self._lines = []

    def _format_call(self, method_name, *args, **kwargs):
        args_str = ', '.join(map(repr, args))
        kwargs_str = ', '.join(f'{k}={v!r}' for k, v in kwargs.items())
        all_args = ', '.join(filter(bool, [args_str, kwargs_str]))
        return f'{self.prefix}.{method_name}({all_args})'

    def new(self, *args, **kwargs):
        call = self._format_call('new', *args, **kwargs)
        self._lines.append(call)
        return self.env.new(*args, **kwargs)

    def set(self, *args, **kwargs):
        call = self._format_call('set', *args, **kwargs)
        self._lines.append(call)
        return self.env.set(*args, **kwargs)

    def new_line(self, *args, **kwargs):
        name = kwargs.get('name')
        call = self._format_call('new_line', *args, **kwargs)
        line = self.env.new_line(*args, **kwargs)

        if name:
            self._lines.append(call)
        else:
            self._temp_lines[line] = call

        return line

    def new_builder(self, *args, **kwargs):
        name = kwargs.get('name')
        printed_components = kwargs.get('components') or args[0]
        for idx, component in enumerate(printed_components):
            if isinstance(component, xt.Line):
                inline_new_line = _Code(self._temp_lines[component])
                printed_components[idx] = inline_new_line
        call = self._format_call('new_builder', *args, **kwargs)
        self._lines.append(f'{name} = {call}')
        builder = self.env.new_builder(*args, **kwargs)
        return EnvWriterProxy(builder, name, parent_proxy=self)

    def place(self, *args, **kwargs):
        name_or_line = kwargs.get('name') or args[0]
        printed_args = list(args)
        printed_kwargs = kwargs.copy()

        if isinstance(name_or_line, xt.Line):
            inline_new_line = _Code(self._temp_lines[name_or_line])
            if printed_kwargs.pop('name', None):
                printed_kwargs['name'] = inline_new_line
            else:
                printed_args[0] = inline_new_line

        call = self._format_call('place', *printed_args, **printed_kwargs)
        self._lines.append(call)
        return self.env.place(*args, **kwargs)

    def build(self):
        self._lines.append(f'{self.prefix}.build()')
        return self.env.build()

    def __setitem__(self, key, value):
        self._lines.append(f'{self.prefix}[{key!r}] = {value!r}')
        self.env[key] = value

    def __getitem__(self, key):
        return self.env[key]

    def save_temp_line(self, line: xt.Line):
        self._temp_lines[line] = f'{self.prefix}.lines[{line.name!r}]'

    @property
    def vars(self):
        class VarsProxy:
            def update(_self, updated):
                self.env.vars.update(updated)
                self._lines.append(f'{self.prefix}.vars.update({updated!r})')

            def __setitem__(_self, key, value):
                self.env.vars[key] = value
                self._lines.append(f'{self.prefix}.vars[{key!r}] = {value!r}')

        return VarsProxy()

    @property
    def _xdeps_vref(self):
        return self.env._xdeps_vref

    @property
    def _xdeps_eref(self):
        return self.env._xdeps_eref

    @property
    def _xdeps_fref(self):
        return self.env._xdeps_fref

    def __str__(self):
        return self.prefix

    def to_string(self):
        preamble = [
            'import xtrack as xt',
            'env = xt.get_environment()',
            'env._xdeps_vref._owner.default_factory = lambda: 0'
        ]
        epilogue = [
            'env._xdeps_vref._owner.default_factory = None',
        ]
        return '\n'.join(preamble + self._lines + epilogue)

    def to_file(self, path: PathLike):
        with open(path, 'w') as f:
            f.write(self.to_string())
