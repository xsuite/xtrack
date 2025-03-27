import io
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict, Union, Literal

from lark import Lark, Transformer, v_args, Token

from xdeps.refs import BaseRef, is_ref

grammar = Path(__file__).with_name('madx.lark').read_text()


# The following types are used to define the output of the parser, to make it
# easier to understand the structure of the parsed MAD-X file and to help
# in spotting mistakes.
VarValueType = Union[int, float, str, bool]


VarType = Union[VarValueType, BaseRef]


class ModifiersType(TypedDict, total=False):
    _repeat: int
    _invert: bool


class ElementType(TypedDict, ModifiersType):
    parent: str
    __extra_items__: Dict[str, VarType]  # Not really supported until PEP 728


class LineType(TypedDict, ModifiersType):
    parent: str  # 'sequence' or 'line'
    l: VarType  # optional, but typing.NotRequired is not available in 3.8
    refer: Literal['centre', 'entry']  # ditto
    elements: List[Tuple[str, Union[ElementType, 'LineType']]]


ElementType = Union[TypedDict('ElementType', {'parent': str}), Dict[str, VarType]]

class MadxOutputType(TypedDict):
    elements: Dict[str, ElementType]
    lines: Dict[str, LineType]
    parameters: Dict[str, ElementType]


@dataclass
class Modifiers:
    repeat: int = 1
    invert: bool = False

    def to_dict(self):
        out = {}
        if self.repeat != 1:
            out['_repeat'] = self.repeat
        if self.invert:
            out['_invert'] = True
        return out


def make_op_handler(op_func):
    return staticmethod(op_func)


def warn(warning):
    print(f'Warning: {warning}')


@v_args(inline=True)
class MadxTransformer(Transformer):
    def __init__(self, vars, functions):
        super().__init__()
        self.elements: Dict[str, ElementType] = {}
        self.lines: Dict[str, LineType] = {}
        self.parameters = {}
        self.vars = vars
        self.functions = functions

    def ignored(self, tokens):
        if tokens:
            statement = ' '.join(str(token) for token in tokens.children)
        else:
            statement = ''
        if statement.startswith('return'):
            return
        if statement == '':
            return
        warn(f'Ignoring statement: `{statement}`')

    def assign_defer(self, name, value) -> Tuple[str, VarType]:
        return name.value.lower(), value

    def assign_value(self, name, value) -> Tuple[str, VarType]:
        if is_ref(value):
            value = value._get_value()
        return name.value.lower(), value

    def special_token_name(self, token):
        return token.value.lower()

    def set_var(self, assignment: Tuple[str, VarType]):
        name, value = assignment
        self.vars[name] = value

    def name_atom(self, name):
        field = name.value.lower()
        if self.vars.default_to_zero and field not in self.vars:
            self.vars[field] = 0
        return self.vars[field]

    def number(self, value):
        float_value = float(value)
        return float_value

    def string_literal(self, string):
        return string.value[1:-1].lower()

    def call(self, function, *args):
        return function(*args)

    def function(self, name_token):
        field = name_token.value.lower()
        return self.functions[field]

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

    def sequence(self, name_token, arglist, *clones) -> Tuple[str, LineType]:
        return name_token.value.lower(), {
            'parent': 'sequence',
            **dict(arglist),
            'elements': list(clones),
        }

    def seqedit(self, arglist, *commands) -> Tuple[str, LineType]:
        (param_name, sequence_name), = arglist

        if param_name != 'sequence':
            raise ValueError(f'Unexpected parameter `{param_name}` of `seqedit`')

        if sequence_name['deferred']:
            raise ValueError('Param `sequence` of `seqedit` cannot be deferred.')

        for command in commands:
            name, params = command

            if any(v['deferred'] for v in params.values()):
                raise ValueError(f'Commands in `seqedit` are not supported with deferred params')

            if name == 'install':
                element_name = params.pop('element')
                self.lines[sequence_name]['elements'].append((element_name, params))
            else:
                warn(f'Command {name} with params {params} is ignored')

    def top_level_sequence(self, sequence):
        name, body = sequence
        self.lines[name] = body

    def clone(self, name_token, command_token, arglist) -> Tuple[str, ElementType]:
        args = dict(arglist)
        parent = command_token.value.lower()

        return name_token.value.lower(), {
            'parent': parent,
            **args,
        }

    def top_level_clone(self, clone):
        name, body = clone
        self.elements[name] = body

    def command_stmt(self, command_token, arglist):
        return command_token.value.lower(), dict(arglist)

    def top_level_command(self, command):
        name, arglist = command
        if name not in self.parameters:
            self.parameters[name] = {}
        self.parameters[name].update(arglist)

    def modifiers(self, *args) -> Modifiers:
        arg_values = set(token.value for token in args)
        modifiers = Modifiers()

        if '-' in arg_values:
            modifiers.invert = True
            arg_values.remove('-')

        if arg_values:
            repeat, = arg_values
            modifiers.repeat = int(repeat)

        return modifiers

    def line_element(self, modifiers, line_item) -> Tuple[str, ElementType]:
        name = None
        body = modifiers.to_dict()
        if isinstance(line_item, Token):
            name = line_item.value.lower()
        elif isinstance(line_item, dict):
            body.update(line_item)
        else:
            raise ValueError(f'Unexpected line element: {line_item}')
        return name, body

    def anonymous_line(self, elements):
        return {
            'parent': 'line',
            'elements': elements,
        }

    def line(self, name_token, anonymous_line) -> Tuple[str, LineType]:
        return name_token.value.lower(), anonymous_line

    def top_level_line(self, line):
        name, body = line
        self.lines[name] = body

    def build_list(self, *args):
        return list(args)

    @v_args(inline=False)
    def start(self, _) -> MadxOutputType:
        return {
            'elements': self.elements,
            'lines': self.lines,
            'parameters': self.parameters,
        }

    def op_arrow(self, a, b):
        a, b = a.lower(), b.lower()
        if b == 'l':
            b = 'length'
        return f'{a}->{b}'

    op_lt = staticmethod(operator.lt)
    op_gt = staticmethod(operator.gt)
    op_le = staticmethod(operator.le)
    op_ge = staticmethod(operator.ge)
    op_eq = staticmethod(operator.eq)
    op_ne = staticmethod(operator.ne)
    op_add = staticmethod(operator.add)
    op_sub = staticmethod(operator.sub)
    op_mul = staticmethod(operator.mul)
    op_div = staticmethod(operator.truediv)
    op_pow = staticmethod(operator.pow)
    op_neg = staticmethod(operator.neg)
    op_pos = staticmethod(operator.pos)


class MadxParser:
    def __init__(self, vars, functions):
        self.transformer = MadxTransformer(vars, functions)
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
        yaml_out = yaml.dump(out, Dumper=yaml.SafeDumper, sort_keys=False)

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
