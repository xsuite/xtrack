from copy import deepcopy
from typing import Dict, Optional, List, Set, Tuple, Union


import numpy as np

import xtrack as xt
from xtrack import BeamElement
from xtrack.environment import Builder
from xtrack.mad_parser.env_writer import EnvWriterProxy
from xtrack import Environment
from xtrack.mad_parser.parse import ElementType, LineType, MadxParser, VarType, MadxOutputType

EXTRA_PARAMS = {
    "slot_id",
    "mech_sep",
    "assembly_id",
    "kmin",
    "kmax",
    "calib",
    "polarity",
    "aper_tol",
}

TRANSLATE_PARAMS = {
    "l": "length",
    "lrad": "length",
    "tilt": "rot_s_rad",
    "from": "from_",
    "e1": "edge_entry_angle",
    "e2": "edge_exit_angle",
    "fint": "edge_entry_fint",
    "fintx": "edge_exit_fint",
}

REVERSED_SUFFIX = '_reversed'


def _warn(msg):
    print(f'Warning: {msg}')


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
    return f'{name}{REVERSED_SUFFIX}'


def _invert(value: Union[str, int, float]):
    if isinstance(value, str):
        return f'-({value})'
    return -value


class MadxLoader:
    def __init__(
            self,
            reverse_lines: Optional[List[str]] = None,
            env: Union[xt.Environment, EnvWriterProxy] = None,
    ):
        self.reverse_lines = reverse_lines or []

        self._madx_elem_hierarchy: Dict[str, List[str]] = {}
        self._reversed_elements: Set[str] = set()
        self._both_direction_elements: Set[str] = set()
        self._builtin_types = set()
        self._parameter_cache = {}

        self.rbarc = True

        self.env = env or xt.Environment()

        self._init_environment()

    def _init_environment(self):
        self.env.vars.default_to_zero = True

        # Define the builtin MAD-X variables
        self.env.vars.update({
            "pi": np.pi,
            "twopi": np.pi * 2,
            "degrad": 180 / np.pi,  # deg/rad
            "raddeg": np.pi / 180,  # rad/deg
            "e": np.e,
            "emass": 0.51099895000e-3,  # GeV
            "pmass": 0.93827208816,  # GeV
            "nmass": 0.93956542052,  # GeV
            "umass": 0.93149410242,  # GeV
            "mumass": 0.1056583715,  # GeV
            "clight": 299792458.0,  # m/s
            "qelect": 1.602176634e-19,  # A * s
            "hbar": 6.582119569e-25,  # MeV * s
            "erad": 2.8179403262e-15,  # m
            "prad": 'erad / emass * pmass',
        })

        # Define the built-in MAD-X elements
        self._new_builtin("vkicker", "Multipole")
        self._new_builtin("hkicker", "Multipole")
        self._new_builtin("tkicker", "Multipole")
        self._new_builtin("kicker", "Multipole")
        self._new_builtin("drift", "Drift")
        self._new_builtin("collimator", "Drift")
        self._new_builtin("rcollimator", "Drift")
        self._new_builtin("ecollimator", "Drift")
        self._new_builtin("instrument", "Drift")
        self._new_builtin("monitor", "Drift")
        self._new_builtin("hmonitor", "Drift")
        self._new_builtin("vmonitor", "Drift")
        self._new_builtin("placeholder", "Drift")
        self._new_builtin("sbend", "Bend")  # no rbarc since we don't have an angle
        self._new_builtin("rbend", "Bend", rbend=True)
        self._new_builtin("quadrupole", "Quadrupole")
        self._new_builtin("sextupole", "Sextupole")
        self._new_builtin("octupole", "Octupole")
        self._new_builtin("marker", "Marker")
        self._new_builtin("rfcavity", "Cavity")
        self._new_builtin("multipole", "Multipole", knl=6 * [0])
        self._new_builtin("solenoid", "Solenoid")
        self._new_builtin('circle', 'LimitEllipse')
        self._new_builtin('ellipse', 'LimitEllipse')
        self._new_builtin('rectangle', 'LimitRect')
        self._new_builtin("rectellipse", 'LimitRectEllipse')
        self._new_builtin("racetrack", 'LimitRacetrack')

    def load_file(self, file, build=True) -> Optional[List[Builder]]:
        """Load a MAD-X file and generate/update the environment."""
        parser = MadxParser()
        parsed_dict = parser.parse_file(file)
        return self.load_parsed_dict(parsed_dict, build=build)

    def load_string(self, string, build=True) -> Optional[List[Builder]]:
        """Load a MAD-X string and generate/update the environment."""
        parser = MadxParser()
        parsed_dict = parser.parse_string(string)
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
            self._new_element(name, parent, self.env, **params, extra=extras)

    def _parse_lines(self, lines: Dict[str, LineType], build=True) -> List[Builder]:
        builders = []

        for name, line_params in lines.items():
            params = line_params.copy()
            line_type = params.pop('parent')
            self._madx_elem_hierarchy[name] = [line_type]

            if line_type == 'sequence':
                refer = params.get('refer', {}).get('expr', 'centre')
                builder = self.env.new_builder(name=name, refer=refer)
                self._parse_components(builder, params.pop('elements'))
                builders.append(builder)
            elif line_type == 'line':
                components = self._parse_line_components(params.pop('elements'))
                builder = self.env.new_builder(name=name, components=components)
            else:
                raise ValueError(
                    'Only a MAD-X sequence or a line type can be used to build'
                    'a line, but got: {line_type}!'
                )

            if build:
                builder.build()

        return builders

    def _parse_parameters(self, parameters: Dict[str, Dict[str, str]]):
        for element, el_params in parameters.items():
            params, extras = get_params(el_params, parent=element)
            self._set_element(element, self.env, **params, extra=extras)

    def _parse_components(self, builder, elements: List[Tuple[str, Union[ElementType, LineType]]]):
        for name, element in elements:
            params = element.copy()
            parent = params.pop('parent', None)
            assert parent != 'sequence'
            params, extras = get_params(params, parent=parent)
            self._new_element(name, parent, builder, **params, extra=extras)

    def _parse_line_components(self, elements):
        components = []

        for name, body in elements:
            # Parent is None if the element already exists and is referred to,
            # by name, otherwise we expect a line nested in the current one.
            parent = body.get('parent', None)
            repeat = body.get('_repeat', 1)
            invert = body.get('_invert', False)
            instance = self.env[name] if name else None

            if parent is None and isinstance(instance, xt.Line):
                # If it's a line, we use __mul__ and __neg__ directly
                element = instance
                if invert:
                    element = -element
                element = repeat * element
                components.append(element)
            elif parent == 'line':
                # If it's a nested line, we parse it recursively
                element = self._parse_line_components(body['elements'])
                if invert:
                    element = list(reversed(element))
                element = repeat * element
                components += element
            elif parent is None:
                # If it's a reference to a single element, we multiply it and
                # add it. Reversal will not affect it.
                components += body.get('_repeat', 1) * [name]
            else:
                raise ValueError('Only an element reference or a line is accepted')

        return components

    def _new_builtin(self, name, xt_type, **kwargs):
        self.env.new(name, xt_type, **kwargs)
        self._builtin_types.add(name)

    def _new_element(self, name, parent, builder, **kwargs):
        self._parameter_cache[name] = {}
        self._parameter_cache[name].update(self._parameter_cache.get(parent, {}))
        self._parameter_cache[name].update(kwargs)

        el_params = self._pre_process_element_params(name, kwargs)
        length = el_params.get('length', self._parameter_cache[name].get('length', 0))
        if self._mad_base_type(name) in {'vkicker', 'hkicker', 'kicker', 'tkicker', 'multipole'}:
            # Workaround for the fact that Multipole.length does not make an element thick
            length = 0

        if parent is None:
            # If parent is None, we wish to place instead
            if self._mad_base_type(name) == 'rbend':
                el_params.pop('rbend', None)
                el_params.pop('rbarc', None)
                el_params.pop('k0_from_h', None)

            if (superfluous := el_params.keys() - {'at', 'from_', 'extra'}):
                raise ValueError(
                    f'Cannot place the element `{name}` as it overrides the '
                    f'parameters: {superfluous}!'
                )

            if (extras := el_params.pop('extra', None)):
                _warn(f'Ignoring extra parameters {extras} for element `{name}`!')


            if (isinstance(self.env[name], BeamElement) and not self.env[name].isthick
                 and length and not isinstance(self.env[name], xt.Marker)):
                drift_name = f'drift_{name}'
                self.env.new(drift_name, 'Drift', length=f'({length}) / 2')
                name = builder.new_line([drift_name, name, drift_name])
            builder.place(name, **el_params)
        else:
            if (isinstance(self.env[parent], BeamElement) and not self.env[parent].isthick
                and length and not isinstance(self.env[parent], xt.Marker)):
                drift_name = f'{name}_drift'
                self.env.new(drift_name, 'Drift', length=f'({length}) / 2')
                at, from_ = el_params.pop('at', None), el_params.pop('from_', None)
                self.env.new(name, parent, **el_params)
                name = self.env.new_line([drift_name, name, drift_name])
                builder.place(name, at=at, from_=from_)
            else:
                if name == parent:
                    el_params.pop('extra', None)
                    builder.place(name, **el_params)
                else:
                    builder.new(name, parent, **el_params)

    def _set_element(self, name, builder, **kwargs):
        self._parameter_cache[name].update(kwargs)
        el_params = self._pre_process_element_params(name, kwargs)
        el_params.pop('from_', None)
        el_params.pop('at', None)
        builder.set(name, **el_params)

    def _pre_process_element_params(self, name, params):
        parent_name = self._mad_base_type(name)

        if parent_name in {'sbend', 'rbend'}:
            # Because of the difficulty in handling the angle (it's not part of
            # the element definition), making an rbend from another rbend is
            # broken. We give up, and just cache all the parameters from the
            # hierarchy – that way `_handle_bend_kwargs` always gets the full
            # picture.
            params = self._parameter_cache[name]

            params['rbend'] = parent_name == 'rbend'

            # `_handle_bend_kwargs` errors if there is rbarc but no angle
            params['rbarc'] = self.rbarc if 'angle' in params else False

            # Default MAD-X behaviour is to take k0 from h only if k0 is not
            # given.
            if 'k0' not in params:
                params['k0_from_h'] = True

            length = params.get('length', 0)
            if (k2 := params.pop('k2', None)) and length:
                params['knl'] = [0, 0, f'({k2}) * ({length})']
            if (k1s := params.pop('k1s', None)) and length:
                params['ksl'] = [0, f'({k1s}) * ({length})']
            if (hgap := params.pop('hgap', None)):
                params['edge_entry_hgap'] = hgap
                params['edge_exit_hgap'] = hgap

        elif parent_name in {'rfcavity', 'rfmultipole'}:
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

        elif parent_name == 'multipole':
            if (knl := params.pop('knl', None)):
                params['knl'] = knl
            if (ksl := params.pop('ksl', None)):
                params['ksl'] = ksl
            if params.pop('lrad', None):
                _warn(f'Multipole `{name}` was specified with a length, ignoring!')

        elif parent_name == 'vkicker':
            if (kick := params.pop('kick', None)):
                params['ksl'] = [kick]

        elif parent_name == 'hkicker':
            if (kick := params.pop('kick', None)):
                params['knl'] = [f'-({kick})']

        elif parent_name in {'kicker', 'tkicker'}:
            if (vkick := params.pop('vkick', None)):
                params['ksl'] = [vkick]
            if (hkick := params.pop('hkick', None)):
                params['knl'] = [f'-({hkick})']

        elif parent_name == 'circle':
            if (aperture := params.pop('aperture', None)):
                params['a'] = params['b'] = aperture[0]
                params['rot_s_rad'] = params.get('aper_tilt', 0)
                # aper_offset = params.get('aper_offset', [0, 0])
                # params['shift_x'] = aper_offset[0]
                # params['shift_y'] = aper_offset[1]

        elif parent_name == 'ellipse':
            if (aperture := params.pop('aperture', None)):
                params['a'] = aperture[0]
                params['b'] = aperture[1]
                params['rot_s_rad'] = params.get('aper_tilt', 0)

        elif parent_name == 'rectangle':
            if (aperture := params.pop('aperture', None)):
                params['min_x'] = -aperture[0]
                params['max_x'] = aperture[0]
                params['min_y'] = -aperture[1]
                params['max_y'] = aperture[1]
                params['rot_s_rad'] = params.get('aper_tilt', 0)
                # params['shift_x'] = params.get('v_pos', 0)

        elif parent_name == 'rectellipse':
            if (aperture := params.pop('aperture', None)):
                params['max_x'] = aperture[0]
                params['max_y'] = aperture[1]
                params['a'] = aperture[2]
                params['b'] = aperture[3]
                params['rot_s_rad'] = params.get('aper_tilt', 0)
                # params['shift_x'] = params.get('v_pos', 0)

        elif parent_name == 'racetrack':
            if (aperture := params.pop('aperture', None)):
                params['min_x'] = -aperture[0]
                params['max_x'] = aperture[0]
                params['min_y'] = -aperture[1]
                params['max_y'] = aperture[1]
                params['a'] = aperture[2]
                params['b'] = aperture[3]
                params['rot_s_rad'] = params.get('aper_tilt', 0)

        if 'edge_entry_fint' in params and 'edge_exit_fint' not in params:
            params['edge_exit_fint'] = params['edge_entry_fint']
            # TODO: Technically MAD-X behaviour is that if edge_exit_fint < 0
            #  then we take edge_entry_fint as edge_exit_fint. But also,
            #  edge_entry_fint is the default value for edge_exit_fint.
            #  To implement this (unhinged?) feature faithfully we'd need to
            #  evaluate the expression here and ideally have a dynamic if-then
            #  expression... Instead, let's just pretend that edge_exit_fint
            #  should be taken as is, and hope no one relies on it being < 0.

        if params.pop('aperture', None):
            pass
            # Avoid flooding the user with warnings
            # _warn(f'Ignoring aperture parameter for element `{name}` for now. '
            #       f'Only apertures on markers and standalone aperture elements '
            #       f'are supported for now.')

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

            new_elements = []
            line_elements = line_params['elements']
            for name, elem_params in reversed(line_elements):
                assert elem_params.get('parent', None) != 'sequence', 'Nesting not yet supported!'
                el = self._reverse_element(name, elem_params, line_params.get('l'))
                new_elements.append((_reversed_name(name), el))

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
            if line_params['parent'] == 'line':
                return

            for name, elem_params in line_params['elements']:
                parent = elem_params.get('parent', name)
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
            if line_params['parent'] == 'line':
                return

            for name, elem_params in line_params['elements']:
                if (parent := elem_params.get('parent', None)):
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
                element[key]['expr'] = _invert(element[key]["expr"])

        def _exchange_fields(key1, key2):
            value1 = element.pop(key1, None)
            value2 = element.pop(key2, None)

            if value1 is not None:
                element[key2] = value1

            if value2 is not None:
                element[key1] = value2


        _reverse_field('k0s')
        _reverse_field('k1')
        _reverse_field('k2s')
        _reverse_field('k3')
        _reverse_field('ks')
        _reverse_field('ksi')
        _reverse_field('vkick')
        _reverse_field('tilt')

        if self._mad_base_type(name) == 'vkicker':
            _reverse_field('kick')

        if 'lag' in element:
            element['lag']['expr'] = f'0.5 - ({element["lag"]["expr"]})'

        if 'at' in element:
            if 'from' in element:
                element['at']['expr'] = _invert(element["at"]["expr"])
                element['from']['expr'] = _reversed_name(element['from']['expr'])
            else:
                if not line_length:
                    raise ValueError(
                        f'Line length must be specified when reversing, however '
                        f'got no length for `{name}`!'
                    )
                element['at']['expr'] = f'({line_length["expr"]}) - ({element["at"]["expr"]})'

        if 'knl' in element:
            knl = element['knl']['expr']
            for i in range(1, len(knl), 2):
                knl[i] = _invert(knl[i])

        if 'ksl' in element:
            ksl = element['ksl']['expr']
            for i in range(0, len(ksl), 2):
                ksl[i] = _invert(ksl[i])

        parent_name = element.get('parent')
        if parent_name and parent_name != self._mad_base_type(parent_name):
            element['parent'] = _reversed_name(parent_name)

        _exchange_fields('e1', 'e2')
        _exchange_fields('h1', 'h2')

        if not ('fint' in element and 'fintx' not in element):
            _exchange_fields('fint', 'fintx')

        return element

    def _mad_base_type(self, element_name: str):
        if element_name.endswith(REVERSED_SUFFIX):
            element_name = element_name[:-len(REVERSED_SUFFIX)]

        if element_name in self._madx_elem_hierarchy:
            return self._madx_elem_hierarchy[element_name][-1]

        if element_name not in self._builtin_types:
            raise ValueError(
                f'Something went wrong: cannot identify the MAD-X base type of'
                f'element `{element_name}`!'
            )

        return element_name

def load_madx_lattice(file, reverse_lines=None):
    loader = MadxLoader(reverse_lines=reverse_lines)
    loader.load_file(file)
    env = loader.env
    return env