import io
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict, Union

from lark import Lark, Transformer, v_args

grammar = Path(__file__).with_name('madx.lark').read_text()


# The following types are used to define the output of the parser, to make it
# easier to understand the structure of the parsed MAD-X file and to help
# in spotting mistakes.
VarValueType = Union[int, float, str, bool]

class VarType(TypedDict):
    expr: Union[VarValueType, List[VarValueType]]
    deferred: bool


ElementType = Union[TypedDict('ElementType', {'parent': str}), Dict[str, VarType]]

class LineType(TypedDict):
    parent: str
    l: VarType
    elements: Dict[str, Union[ElementType, 'LineType']]

class MadxOutputType(TypedDict):
    vars: Dict[str, VarType]
    elements: Dict[str, ElementType]
    lines: Dict[str, LineType]
    parameters: Dict[str, ElementType]


def make_op_handler(op):
    def op_func(a, b):
        return f'({a} {op} {b})'
    return staticmethod(op_func)


def warn(warning):
    print(f'Warning: {warning}')


@v_args(inline=True)
class MadxTransformer(Transformer):
    def __init__(self):
        super().__init__()
        self.vars: Dict[str, VarType] = {}
        self.elements: Dict[str, ElementType] = {}
        self.lines: Dict[str, LineType] = {}
        self.parameters = {}

    def ignored(self, line):
        warn(f'Ignoring line: {line}')

    def assign_defer(self, name, value) -> Tuple[str, VarType]:
        return name.value.lower(), {
            'expr': value,
            'deferred': True,
        }

    def assign_value(self, name, value) -> Tuple[str, VarType]:
        return name.value.lower(), {
            'expr': value,
            'deferred': False,
        }

    def set_var(self, assignment: Tuple[str, VarType]):
        name, value = assignment
        self.vars[name] = value

    def name_atom(self, name):
        field = name.value
        return field.lower()

    def constant(self, const_token):
        return const_token.value.lower()

    def number(self, value):
        float_value = float(value)
        return float_value

    def call(self, function, *args):
        return f'{function}({", ".join(args)})'

    def function(self, name_token):
        return name_token.value.lower()

    def command(self, name_token, arglist):
        command = name_token.value
        arglist = arglist if isinstance(arglist, list) else [arglist]
        return command.lower(), arglist

    @v_args(inline=False)
    def command_arglist(self, args):
        return args

    def set_flag(self, name_token):
        return name_token.value.lower(), True

    def reset_flag(self, name_token):
        return name_token.value.lower(), False

    def sequence(self, name_token, arglist, *clones):
        return name_token.value.lower(), {
            'parent': 'sequence',
            **dict(arglist),
            'elements': dict(clones),
        }

    def top_level_sequence(self, sequence):
        name, body = sequence
        self.lines[name] = body

    def clone(self, name_token, command_token, arglist):
        return name_token.value.lower(), {
            'parent': command_token.value.lower(),
            **dict(arglist),
        }

    def top_level_clone(self, clone):
        name, body = clone
        self.elements[name] = {
            'parent': name,
            **body,
        }

    def command_stmt(self, command_token, arglist):
        return command_token.value.lower(), dict(arglist)

    def top_level_command(self, command):
        name, arglist = command
        if name not in self.parameters:
            self.parameters[name] = {}
        self.parameters[name].update(arglist)

    def build_list(self, *args):
        return list(args)

    @v_args(inline=False)
    def start(self, _) -> MadxOutputType:
        return {
            'vars': self.vars,
            'elements': self.elements,
            'lines': self.lines,
            'parameters': self.parameters,
        }

    def get_item(self, lhs, rhs):
        raise NotImplementedError('The `->` syntax is not yet supported')

    op_lt = make_op_handler('<')
    op_gt = make_op_handler('>')
    op_le = make_op_handler('<=')
    op_ge = make_op_handler('>=')
    op_eq = make_op_handler('==')
    op_ne = make_op_handler('!=')
    op_add = make_op_handler('+')
    op_sub = make_op_handler('-')
    op_mul = make_op_handler('*')
    op_div = make_op_handler('/')
    op_pow = make_op_handler('**')

    def op_neg(self, a):
        if isinstance(a, float):
            return -a
        return f'-{a}'

    def op_pos(self, a):
        if isinstance(a, float):
            return a
        return f'+{a}'


class MadxParser:
    def __init__(self):
        self.transformer = MadxTransformer()
        self.parser = Lark(grammar, parser='lalr', transformer=self.transformer)

    def parse_string(self, text: str) -> MadxOutputType:
        return self.parser.parse(text)  # noqa

    def parse_file(self, path_or_file: Union[str, Path, io.TextIOBase]) -> MadxOutputType:
        if not isinstance(path_or_file, io.TextIOBase):
            file = open(path_or_file)
        else:
            file = path_or_file

        with file:
            return self.parse_string(file.read())


# Run as a script to be able to inspect a MAD-X file as YAML
if __name__ == '__main__':
    import click
    import time
    import yaml

    @click.command()
    @click.argument('file_name')
    @click.option(
        '--output', '-o',
        help='Path to the output file (default: same as input but ending with `.yaml`)',
        default=None,
    )
    @click.option('--test', '-t', is_flag=True, help='Try to load the sequence into Xsuite', default=False)
    def main(file_name, output, test):
        t0 = time.time()
        out = MadxParser().parse_file(file_name)
        print(f"Parsed `{file_name}` in {time.time() - t0} s")

        # This output is for visualisation purposes only: dict ordering is not guaranteed
        # out of the box by the YAML standard (should use !!omap, but it's not supported by PyYAML)
        yaml_out = yaml.dump(out, Dumper=yaml.CDumper, sort_keys=False)

        if not output:
            outfile = Path(file_name).with_suffix(suffix='.yaml')
        else:
            outfile = Path(output)

        with outfile.open('w') as f:
            f.write(yaml_out)

        if test:
            from xtrack.mad_parser.loader import MadxLoader
            loader = MadxLoader()
            loader.load_file(outfile)

    main()
