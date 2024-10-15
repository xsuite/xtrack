from copy import deepcopy
from typing import Dict, Optional, List, Set, Tuple

import numpy as np

import xtrack as xt
from xtrack import Environment
from xtrack.environment import Builder
from xtrack.mad_parser.parse import ElementType, LineType, MadxParser, VarType, MadxOutputType

EXTRA_PARAMS = {
    "slot_id",
    "mech_sep",
    "assembly_id",
    "kmin",
    "kmax",
    "calib",
    "polarity",
}

TRANSLATE_PARAMS = {
    "l": "length",
    "lrad": "length",
    "tilt": "rot_s_rad",
    "from": "from_",
    "e1": "edge_entry_angle",
    "e2": "edge_exit_angle",
}


def get_params(params, parent):
    params = params.copy()
    if parent in {'placeholder', 'instrument'}:
        _ = params.pop('lrad', None)

    def _normalise_single(param):
        lower = param.lower()
        return TRANSLATE_PARAMS.get(lower, lower)

    # TODO: Ignoring value['deferred'] for now.
    normalised = {_normalise_single(k): v['expr'] for k, v in params.items()}

    main_params = {}
    extras = {}
    for k, v in normalised.items():
        if k in EXTRA_PARAMS:
            extras[k] = v
        else:
            main_params[k] = v

    return main_params, extras


def _reversed_name(name: str):
    return f'{name}^'


class MadxLoader:
    def __init__(self, reverse_lines: Optional[List[str]] = None):
        self._madx_elem_hierarchy: Dict[str, str] = {}
        self.reverse_lines = reverse_lines or []

        self.env = xt.Environment()
        self._init_environment()

        self._reversed_elements: Set[str] = set()
        self._both_direction_elements: Set[str] = set()

    def _init_environment(self):
        self.env._xdeps_vref._owner.default_factory = lambda: 0

        # Define the builtin MAD-X variables
        self.env.vars["twopi"] = np.pi * 2

        # Define the built-in MAD-X elements
        self.env.new("vkicker", "Multipole")
        self.env.new("hkicker", "Multipole")
        self.env.new("tkicker", "Multipole")
        self.env.new("collimator", "Drift")
        self.env.new("instrument", "Drift")
        self.env.new("monitor", "Drift")
        self.env.new("placeholder", "Drift")
        self.env.new("sbend", "Bend")
        self.env.new("rbend", "Bend")
        self.env.new("quadrupole", "Quadrupole")
        self.env.new("sextupole", "Sextupole")
        self.env.new("octupole", "Octupole")
        self.env.new("marker", "Drift")
        self.env.new("rfcavity", "Cavity")
        self.env.new("multipole", "Multipole", knl=[0, 0, 0, 0, 0, 0])
        self.env.new("solenoid", "Solenoid")

    def load_file(self, file, build=True) -> Optional[List[Builder]]:
        """Load a MAD-X file and generate/update the environment."""
        parser = MadxParser()
        parsed_dict = parser.parse_file(file)
        return self.load_parsed_dict(parsed_dict, build=build)

    def load_parsed_dict(self, parsed_dict: MadxOutputType, build=True) -> Optional[List[Builder]]:
        hierarchy = self._collect_hierarchy(parsed_dict)
        self._madx_elem_hierarchy.update(hierarchy)

        if self.reverse_lines:
            collected_names = self._collect_reversed_elements(parsed_dict)
            straight_names, reversed_names = collected_names
            self._reversed_elements.update(reversed_names)
            self._both_direction_elements.update(straight_names & reversed_names)
            self._reverse_lines(parsed_dict)

        self._parse_vars(parsed_dict["vars"])
        self._parse_elements(parsed_dict["elements"])
        builders = self._parse_lines(parsed_dict["lines"], build=build)
        self._parse_parameters(parsed_dict["parameters"])

        if not build:
            return builders

    def _parse_vars(self, vars: Dict[str, VarType]):
        for var_name, var_value in vars.items():
            # TODO: Ignoring var_value['deferred'] for now.
            self.env[var_name] = var_value['expr']

    def _parse_elements(self, elements: Dict[str, ElementType]):
        for name, el_params in elements.items():
            parent = el_params.pop('parent')
            assert parent != 'sequence'
            params, extras = get_params(el_params, parent=parent)
            self.env.new(name, parent, **params, extra=extras)

    def _parse_lines(self, lines: Dict[str, LineType], build=True) -> List[Builder]:
        builders = []

        for name, line_params in lines.items():
            params = line_params.copy()
            assert params.pop('parent') == 'sequence'
            builder = self.env.new_builder(name=name)
            self._parse_components(builder, params.pop('elements'))
            builders.append(builder)
            if build:
                builder.build()

        return builders

    def _parse_parameters(self, parameters: Dict[str, Dict[str, str]]):
        for element, el_params in parameters.items():
            params, extras = get_params(el_params, parent=element)
            self._set_element(element, **params, extra=extras)

    def _parse_components(self, builder, elements: Dict[str, LineType]):
        for name, element in elements.items():
            params = element.copy()
            parent = params.pop('parent')
            assert parent != 'sequence'
            params, extras = get_params(params, parent=parent)
            self._new_element(name, parent, builder, **params, extra=extras)

    def _new_element(self, name, parent, builder=None, **kwargs):
        if not builder:
            builder = self.env

        el_params = self._pre_process_element_params(name, kwargs)
        builder.new(name, parent, **el_params)

    def _set_element(self, name, builder=None, **kwargs):
        if not builder:
            builder = self.env
        el_params = self._pre_process_element_params(name, kwargs)
        builder.set(name, **el_params)

    def _pre_process_element_params(self, name, params):
        parent_name = self._mad_base_type(name)

        if parent_name == 'rbend':
            params['rbend'] = True

        if parent_name in {'rfcavity', 'rfmultipole'}:
            if (lag := params.pop('lag', None)):
                params['lag'] = f'({lag}) * 360'
            if (volt := params.pop('volt', None)):
                params['voltage'] = f'({volt}) * 1e6'
            if (freq := params.pop('freq', None)):
                params['frequency'] = f'({freq}) * 1e6'
            if 'harmon' in params:
                # harmon * beam.beta * clight / sequence.length
                # raise NotImplementedError
                pass

        return params

    def _reverse_lines(self, parsed_dict: MadxOutputType):
        """Reverse a line in place, by adding reversed elements where necessary."""
        # Deal with element definitions
        elements_dict = parsed_dict['elements']
        defined_names = list(elements_dict.keys())  # especially here order matters!
        for name in defined_names:
            if name not in self._reversed_elements:
                continue

            element = parsed_dict['elements'][name]
            reversed_element = self._reverse_element(name, element)
            elements_dict[_reversed_name(name)] = reversed_element

            if name not in self._both_direction_elements:
                # We can remove the original element if it's not needed
                del elements_dict[name]

        # Deal with line definitions
        for line_name, line_params in parsed_dict['lines'].items():
            if line_name not in self.reverse_lines:
                continue

            new_elements = {}
            line_elements = line_params['elements']
            for name, elem_params in reversed(line_elements.items()):
                assert elem_params['parent'] != 'sequence', 'Nesting not yet supported!'
                el = self._reverse_element(name, elem_params, line_params.get('l'))
                new_elements[_reversed_name(name)] = el

            line_params['elements'] = new_elements

        # Deal with the parameters
        parametrised_names = list(parsed_dict['parameters'].keys())  # ordered again
        for name in parametrised_names:
            if name not in self._reversed_elements:
                continue

            params = parsed_dict['parameters'][name]
            reversed_params = self._reverse_element(name, params)
            parsed_dict['parameters'][_reversed_name(name)] = reversed_params

            if name not in self._both_direction_elements:
                # We can remove the original parameter setting if it's not needed
                del parsed_dict['parameters'][name]


    def _collect_reversed_elements(self, parsed_dict: MadxOutputType) -> Tuple[Set[str], Set[str]]:
        """Collect elements that are shared between non- and reversed lines."""
        straight: Set[str] = set()
        reversed_: Set[str] = set()

        def _descend_into_line(line_params, correct_set):
            for name, elem_params in line_params['elements'].items():
                parent = elem_params['parent']
                correct_set.add(name)
                # Also add the chain of parent types, as they will also need to
                # be reversed. We skip the last element, as it's the base madx
                # type and so it's empty (nothing to reverse).
                for parent_name in self._madx_elem_hierarchy[name][:-1]:
                    correct_set.add(parent_name)
                if parent == 'sequence':
                    _descend_into_line(elem_params, correct_set)

        for line_name, line_params in parsed_dict["lines"].items():
            correct_set = reversed_ if line_name in self.reverse_lines else straight
            _descend_into_line(line_params, correct_set)

        return straight, reversed_

    def _collect_hierarchy(self, parsed_dict: MadxOutputType):
        """Collect the base Madx types of all defined elements."""
        hierarchy = {}

        def _descend_into_line(line_params):
            for name, elem_params in line_params['elements'].items():
                parent = elem_params['parent']
                hierarchy[name] = [parent] + hierarchy.get(parent, [])
                if parent == 'sequence':
                    _descend_into_line(elem_params)

        for name, el_params in parsed_dict['elements'].items():
            parent = el_params['parent']
            hierarchy[name] = [parent] + hierarchy.get(parent, [])

        for line_name, line_params in parsed_dict['lines'].items():
            _descend_into_line(line_params)

        return hierarchy

    def _reverse_element(self, name: str, element: ElementType, line_length: Optional[VarType] = None) -> ElementType:
        """Return a reversed element without modifying the original."""
        element = deepcopy(element)

        UNSUPPORTED = {
            'dipedge', 'wire', 'crabcavity', 'rfmultipole', 'beambeam',
            'matrix', 'srotation', 'xrotation', 'yrotation', 'translation',
            'nllens',
        }

        if (type_ := self._mad_base_type(name)) in UNSUPPORTED:
            raise NotImplementedError(
                f'Cannot reverse the element `{name}`, as reversing elements '
                f'of type `{type_}` is not supported!'
            )

        def _reverse_field(key):
            if key in element:
                element[key]['expr'] = f'-({element[key]["expr"]})'

        _reverse_field('k0s')
        _reverse_field('k1')
        _reverse_field('k2s')
        _reverse_field('k3')
        _reverse_field('ks')
        _reverse_field('ksi')
        _reverse_field('vkick')

        if 'lag' in element:
            element['lag']['expr'] = f'180 - ({element["lag"]["expr"]})'

        if 'at' in element:
            if 'from' in element:
                element['at']['expr'] = f'-({element["at"]["expr"]})'
                element['from']['expr'] = _reversed_name(element['from']['expr'])
            else:
                if not line_length:
                    raise ValueError(
                        f'Line length must be specified when reversing, however '
                        f'got no length for `{name}`!'
                    )
                element['at']['expr'] = f'({line_length["expr"]}) - ({element["at"]["expr"]})'

        e1, e2 = element.get('e1', 0), element.get('e2', 0)
        if e1 != e2:
            element['e1'], element['e2'] = e2, e1

        if 'knl' in element:
            knl = element['knl']['expr'].copy()
            for i, knli in enumerate(knl[1::2]):
                knl[i] = f'-({knli})'
            element['knl']['expr'] = knl

        if 'ksl' in element:
            ksl = element['ksl']['expr'].copy()
            for i, ksli in enumerate(ksl[0::2]):
                ksl[i] = f'-({ksli})'
            element['ksl']['expr'] = ksl

        parent_name = element.get('parent')
        if parent_name and parent_name != self._mad_base_type(parent_name):
            element['parent'] = _reversed_name(parent_name)

        return element

    def _mad_base_type(self, element_name: str):
        if element_name.endswith('^'):
            element_name = element_name[:-1]

        if element_name in self._madx_elem_hierarchy:
            parent_name = self._madx_elem_hierarchy[element_name][-1]
        else:
            parent_name = element_name

        if parent_name in {
            "vkicker",
            "hkicker",
            "tkicker",
            "collimator",
            "instrument",
            "monitor",
            "placeholder",
            "sbend",
            "rbend",
            "quadrupole",
            "sextupole",
            "octupole",
            "marker",
            "rfcavity",
            "multipole",
            "solenoid",
        }:
            return parent_name

        return f'{parent_name}^'
