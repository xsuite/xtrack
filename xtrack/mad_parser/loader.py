from typing import Dict, Optional, List, Set, Tuple, Union

import numpy as np

import xtrack as xt
from xtrack import BeamElement
from xtrack.environment import Builder
from xtrack.mad_parser.parse import ElementType, LineType, MadxParser, VarType, MadxOutputType
from xtrack.environment import _reverse_element

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

CONSTANTS = {
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
}

_APERTURE_TYPES = {
    'circle': 'LimitEllipse',
    'ellipse': 'LimitEllipse',
    'rectangle': 'LimitRect',
    'rectellipse': 'LimitRectEllipse',
    'racetrack': 'LimitRacetrack',
    'octagon': 'LimitPolygon',
    'polygon': 'LimitPolygon',  # not really an explicit option in MAD-X
}


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
    normalised = {_normalise_single(k): v for k, v in params.items()}

    main_params = {}
    extras = {}
    for k, v in normalised.items():
        if k in EXTRA_PARAMS:
            extras[k] = v
        else:
            main_params[k] = v

    return main_params, extras


class MadxLoader:
    def __init__(
            self,
            env: xt.Environment = None,
            default_to_zero: bool = False,
    ):
        self._madx_elem_hierarchy: Dict[str, List[str]] = {}
        self._both_direction_elements: Set[str] = set()
        self._builtin_types = set()
        self._parameter_cache = {}

        self.rbarc = True

        self.env = env or xt.Environment()
        self.env.default_to_zero = default_to_zero

        self._init_environment()

    def _init_environment(self):
        self.env.vars.default_to_zero = True

        # Define the builtin MAD-X variables
        self.env.vars.update(CONSTANTS)

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
        self._new_builtin("sbend", "Bend")
        self._new_builtin("rbend", "RBend")
        self._new_builtin("quadrupole", "Quadrupole")
        self._new_builtin("sextupole", "Sextupole")
        self._new_builtin("octupole", "Octupole")
        self._new_builtin("marker", "Marker")
        self._new_builtin("rfcavity", "Cavity")
        self._new_builtin("multipole", "Multipole", knl=6 * [0])
        self._new_builtin("solenoid", "Solenoid")

        for mad_apertype in _APERTURE_TYPES:
            self._new_builtin(mad_apertype, 'Marker')

    def load_file(self, file, build=True) -> Optional[List[Builder]]:
        """Load a MAD-X file and generate/update the environment."""
        parser = MadxParser(vars=self.env.vars, functions=self.env.functions)
        parsed_dict = parser.parse_file(file)
        return self.load_parsed_dict(parsed_dict, build=build)

    def load_string(self, string, build=True) -> Optional[List[Builder]]:
        """Load a MAD-X string and generate/update the environment."""
        parser = MadxParser(vars=self.env.vars, functions=self.env.functions)
        parsed_dict = parser.parse_string(string)
        return self.load_parsed_dict(parsed_dict, build=build)

    def load_parsed_dict(self, parsed_dict: MadxOutputType, build=True) -> Optional[List[Builder]]:
        hierarchy = self._collect_hierarchy(parsed_dict)
        self._madx_elem_hierarchy.update(hierarchy)

        self._parse_elements(parsed_dict["elements"])
        builders = self._parse_lines(parsed_dict["lines"], build=build)
        self._parse_parameters(parsed_dict["parameters"])

        if not build:
            return builders

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
                refer = params.get('refer', 'centre')
                if refer == 'entry':
                    refer = 'start'
                elif refer == 'exit':
                    refer = 'end'
                length = params.get('l', None)
                builder = self.env.new_builder(name=name, refer=refer,
                                               length=length)
                self._parse_components(builder, params.pop('elements'))
                builders.append(builder)
            elif line_type == 'line':
                components = self._parse_line_components(params.pop('elements'))
                builder = self.env.new_builder(name=name, components=components)
            else:
                raise ValueError(
                    f'Only a MAD-X sequence or a line type can be used to build'
                    f'a line, but got: {line_type}!'
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
        if name not in self.env.element_dict:
            self.env.new(name, xt_type, **kwargs)
        self._builtin_types.add(name)

    def _new_element(self, name, parent, builder, **kwargs):
        should_clone = parent is not None

        if should_clone:
            self._parameter_cache[name] = self._parameter_cache.get(parent, {}).copy()
        else:
            self._parameter_cache[name] = self._parameter_cache.get(name, {})
        self._parameter_cache[name].update(kwargs)

        aperture = self._build_aperture(name, f'{name}_aper', kwargs)
        if not aperture and name in self.env.element_dict:
            aperture = self.env[name].name_associated_aperture
        if not aperture and parent:
            aperture = self.env[parent].name_associated_aperture

        el_params = self._convert_element_params(name, kwargs)

        if aperture and 'at' in el_params:  # placing mode
            builder.place(aperture, at=0, from_=f'{name}@start')

        if should_clone:
            self._clone_element(name, parent, builder, el_params)
        else:
            # If parent is None, we must be in a sequence, and so we are
            # placing the element: in MAD-X this requires an `at` param, but
            # we can be a bit more lax, as Xsuite will automatically place the
            # element after the previous one if there is not `at`.
            self._place_element(name, el_params, builder)

        if aperture:
            builder.element_dict[name].name_associated_aperture = aperture

    def _place_element(self, name, el_params, builder):
        """Place an element in the sequence.

        This is the case when `parent` is None.
        """
        if self._mad_base_type(name) in ['rbend', 'sbend']:
            el_params.pop('k0_from_h', None)

        if (superfluous := el_params.keys() - {'at', 'from_', 'extra'}):
            raise ValueError(
                f'Cannot place the element `{name}` as it overrides the '
                f'parameters: {superfluous}!'
            )

        if (extras := el_params.pop('extra', None)):
            _warn(f'Ignoring extra parameters {extras} for element `{name}`!')
        element = self.env[name]

        length = self._element_length(name, el_params)
        is_not_thick = isinstance(element, BeamElement) and not element.isthick
        if is_not_thick and length and not isinstance(element, xt.Marker):
            # Handle the thin elements that have a length in MAD-X: sandwich
            line = self._make_thick_sandwich(name, length)
            builder.place(line, **el_params)
        else:
            builder.place(name, **el_params)

    def _clone_element(self, name, parent, builder, el_params):
        """Clone an element, and possibly place it if we are in a sequence.

        Here `parent` is not None.
        """
        length = self._element_length(name, el_params)
        element = self.env[parent]
        is_not_thick = isinstance(element, BeamElement) and not element.isthick

        if is_not_thick and length and not isinstance(element, xt.Marker):
            # Handle the thin elements that have a length in MAD-X: sandwich
            at, from_ = el_params.pop('at', None), el_params.pop('from_', None)
            if name not in self.env.element_dict:
                make_drifts = True
                self.env.new(name, parent, **el_params)
            else:
                make_drifts = False
                _warn(f'Element `{name}` already exists, this definition '
                      f'will be ignored (for compatibility with MAD-X)')
            name = self._make_thick_sandwich(name, length, make_drifts)
            builder.place(name, at=at, from_=from_)
        elif name == parent:
            # This happens when the element is cloned with the same name, which
            # is allowed inside MAD-X sequences, e.g.: `el: el, at = 42;`.
            # We cannot attach extra to a place though, so this is skipped.
            if dropped_extra := el_params.pop('extra', None):
                _warn(f'Ignoring extra parameters {dropped_extra} for element '
                      f'`{name}`: it is a clone of itself overriding `extra`.')
            el_params.pop('k0_from_h', None)
            builder.place(name, **el_params)
        else:
            # `force=True` is needed to overwrite existing elements. In MAD-X
            # when an element name is repeated between lines, the first one
            # is retained: we do not simulate this behaviour here.
            builder.new(name, parent, force=True, **el_params)

    def _element_length(self, name, el_params):
        """Given the definition and params of the element, return its length."""
        length = el_params.get('length', self._parameter_cache[name].get('length', 0))
        THIN_ELEMENTS = {'vkicker', 'hkicker', 'kicker', 'tkicker', 'multipole'}
        if self._mad_base_type(name) in THIN_ELEMENTS:
            # Workaround for the elements that are thin despite having a
            # ``length`` parameter.
            length = 0

        return length

    def _make_thick_sandwich(self, name, length, make_drifts=True):
        """Make a sandwich of two drifts around the element."""
        drift_name = f'{name}_drift'
        if make_drifts:
            self.env.new(drift_name + '_0', 'Drift', length=length / 2)
            self.env.new(drift_name + '_1', 'Drift', length=length / 2)
        line = self.env.new_line([drift_name + '_0', name, drift_name + '_1'])
        return line

    def _set_element(self, name, builder, **kwargs):
        self._parameter_cache[name].update(kwargs)

        if 'aperture' in kwargs and 'apertype' not in kwargs:
            kwargs['apertype'] = self._parameter_cache[name]['apertype']
        aperture = self._build_aperture(name, f'{name}_aper', kwargs, force=True)

        el_params = self._convert_element_params(name, kwargs)
        builder.set(name, **el_params)
        builder.element_dict[name].name_associated_aperture = aperture

    def _convert_element_params(self, name, params):
        parent_name = self._mad_base_type(name)

        if parent_name in {'sbend', 'rbend'}:
            # We need to keep the rbarc parameter from the parent element.
            # If rbarc = True, then rbend length is the straight length.
            # If rbarc = False, then the length is the arc length, as for sbend.
            length = params.get('length', 0)

            if parent_name == 'rbend':
                rbarc = self._parameter_cache[name].get('rbarc', self.rbarc)
                if rbarc and 'length' in params:
                    params['length_straight'] = params.pop('length')

            # Default MAD-X behaviour is to take k0 from h only if k0 is not
            # given. We need to replicate this behaviour. Ideally we should
            # evaluate expressions here, but that's tricky.
            if self._parameter_cache[name].get('k0', 0) == 0:
                params['k0_from_h'] = True
            else:
                params['k0_from_h'] = False

            if (k2 := params.pop('k2', None)) and length:
                params['knl'] = [0, 0, k2 * length]
            if (k1s := params.pop('k1s', None)) and length:
                params['ksl'] = [0, k1s * length]
            if (hgap := params.pop('hgap', None)):
                params['edge_entry_hgap'] = hgap
                params['edge_exit_hgap'] = hgap

        elif parent_name in {'rfcavity', 'rfmultipole'}:
            if (lag := params.pop('lag', None)):
                params['lag'] = lag * 360
            if (volt := params.pop('volt', None)):
                params['voltage'] = volt * 1e6
            if (freq := params.pop('freq', None)):
                params['frequency'] = freq * 1e6
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
                params['knl'] = [-kick]

        elif parent_name in {'kicker', 'tkicker'}:
            if (vkick := params.pop('vkick', None)):
                params['ksl'] = [vkick]
            if (hkick := params.pop('hkick', None)):
                params['knl'] = [-hkick]

        if 'edge_entry_fint' in params and 'edge_exit_fint' not in params:
            params['edge_exit_fint'] = params['edge_entry_fint']
            # TODO: Technically MAD-X behaviour is that if edge_exit_fint < 0
            #  then we take edge_entry_fint as edge_exit_fint. But also,
            #  edge_entry_fint is the default value for edge_exit_fint.
            #  To implement this (unhinged?) feature faithfully we'd need to
            #  evaluate the expression here and ideally have a dynamic if-then
            #  expression... Instead, let's just pretend that edge_exit_fint
            #  should be taken as is, and hope no one relies on it being < 0.

        return params

    def _build_aperture(self, name, aper_name, params, force=False):
        """Build a Xtrack aperture for element `name` with  `params`.

        Parameters
        ----------
        name : str
            The name of the element for which to build the aperture.
        aper_name : str
            The name of the aperture element to be created.
        params : dict
            The parameters of the element, including the aperture parameters.
        force : bool, optional
            If True, the ``force`` parameter is passed to the builder when
            creating the aperture element: this will overwrite any existing
            aperture element with the same name. If False, an error will be
            raised if an aperture element with the same name already exists.

        Returns
        -------
        The name of the generated aperture element in the environment, or None.

        Notes:
        ------
        Currently supports all the basic MAD-X aperture types, however when
        ``aper_vx`` or ``aper_vy`` are given, the aperture is assumed to be
        simply a polygon, instead of applying the MAD-X logic (testing first for
        a simple shape and then for a polygon).
        """
        if not {'apertype', 'aperture', 'aper_vx', 'aper_vy'} & set(params):
            # No aperture parameters, nothing to do
            return

        if 'aper_vx' in params or 'aper_vy' in params:
            apertype = 'polygon'
            aperture = None
        else:
            apertype = params.pop('apertype', None) or self._mad_base_type(name)
            aperture = params.pop('aperture', None)

        if apertype not in _APERTURE_TYPES:
            raise ValueError(
                f'The aperture type for the element `{name}` (inferred to be '
                f'`{apertype}`) is not recognised.'
            )

        x_offset, y_offset = params.pop('aper_offset', (0, 0))
        if params.pop('aper_tol', None):
            _warn(f'Aperture tolerance (`{name}`) is not supported, ignoring.')
        aper_params = {
            'rot_s_rad': params.pop('aper_tilt', 0),
            'shift_x': x_offset,
            'shift_y': y_offset,
        }

        if apertype == 'circle':
            aper_params['a'] = aper_params['b'] = aperture[0]

        elif apertype == 'ellipse':
            aper_params['a'] = aperture[0]
            aper_params['b'] = aperture[1]

        elif apertype == 'rectangle':
            aper_params['min_x'] = -aperture[0]
            aper_params['max_x'] = aperture[0]
            aper_params['min_y'] = -aperture[1]
            aper_params['max_y'] = aperture[1]

        elif apertype == 'rectellipse':
            aper_params['max_x'] = aperture[0]
            aper_params['max_y'] = aperture[1]
            aper_params['a'] = aperture[2]
            aper_params['b'] = aperture[3]

        elif apertype == 'racetrack':
            aper_params['min_x'] = -aperture[0]
            aper_params['max_x'] = aperture[0]
            aper_params['min_y'] = -aperture[1]
            aper_params['max_y'] = aperture[1]
            aper_params['a'] = aperture[2]
            aper_params['b'] = aperture[3]

        elif apertype == 'octagon':
            # In MAD the octagon is defined with {w/2, h/2, phi_1, phi_2},
            # where w and h are respectively the width and height of the
            # rectangle that circumscribes the octagon, and phi_1 and phi_2
            # are the two angles sustaining the cut corner in the first
            # quadrant, given in radians, and with phi_1 < phi_2.
            half_w, half_h, phi_1, phi_2 = aperture
            y_right_corner = half_w * self.env.functions.tan(phi_1)
            x_top_corner = half_h / self.env.functions.tan(phi_2)
            top_x_vertices = [half_w, x_top_corner, -x_top_corner, -half_w]
            x_vertices = top_x_vertices + top_x_vertices[::-1]
            right_y_vertices = [y_right_corner, half_h, half_h, y_right_corner]
            y_vertices = right_y_vertices + [-y for y in right_y_vertices]
            aper_params['x_vertices'] = x_vertices
            aper_params['y_vertices'] = y_vertices

        elif apertype == 'polygon':
            if (aper_vx := params.pop('aper_vx', None)):
                aper_params['x_vertices'] = aper_vx
            if (aper_vy := params.pop('aper_vy', None)):
                aper_params['y_vertices'] = aper_vy

        return self.env.new(aper_name, _APERTURE_TYPES[apertype], force=force,
                            **aper_params)

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


    def _mad_base_type(self, element_name: str):

        if element_name in self._madx_elem_hierarchy:
            return self._madx_elem_hierarchy[element_name][-1]

        if element_name not in self._builtin_types:
            raise ValueError(
                f'Cannot identify the MAD-X base type of element `{element_name}`!'
            )

        return element_name

    def _is_standalone_aperture(self, element_name: str):
        return self._mad_base_type(element_name) in _APERTURE_TYPES


def load_madx_lattice(file=None, string=None, reverse_lines=None):

    if file is not None and string is not None:
        raise ValueError('Only one of `file` or `string` can be provided!')

    if file is None and string is None:
        raise ValueError('Either `file` or `string` must be provided!')

    loader = MadxLoader()

    if file is not None:
        loader.load_file(file)
    elif string is not None:
        loader.load_string(string)
    else:
        raise ValueError('Something went wrong!')

    env = loader.env

    if reverse_lines:
        print('Reversing lines:', reverse_lines)
        rlines = {}
        for nn in reverse_lines:
            ll = env.lines[nn]
            llr = ll.copy()

            for enn in llr.element_names:
                _reverse_element(llr, enn)

            llr.discard_tracker()
            llr.element_names = llr.element_names[::-1]

            rlines[nn] = llr

        all_lines = {}
        for nn in env.lines.keys():
            if nn in rlines:
                all_lines[nn] = rlines[nn]
            else:
                all_lines[nn] = env.lines[nn]

        new_env = xt.Environment(lines=all_lines)

        # Adapt builders
        for nn in env.lines.keys():
            bb = env.lines[nn].builder.__class__(new_env)
            bb.__dict__.update(env.lines[nn].builder.__dict__)
            bb.env = new_env
            this_rename = new_env.lines[nn]._renamed_elements
            for cc in bb.components:
                cc.name = this_rename.get(cc.name, cc.name)
                cc.from_ = this_rename.get(cc.from_, cc.from_)

            if nn in reverse_lines:
                length = env.lines[nn].get_length()
                bb.components = bb.components[::-1]
                for cc in bb.components:
                    if cc.at is not None:
                        if isinstance(cc.at, str) or isinstance(cc.at, float):
                            if cc.from_ is not None:
                                cc.at = -cc.at
                            else:
                                cc.at = length - cc.at
            new_env.lines[nn].builder = bb

        # Add to new environment elements that were not in any line
        elems_not_in_lines = set(env.elements.keys())
        for nn in env.lines.keys():
            elems_not_in_lines -= set(env.lines[nn].element_names)
        dummy_line = env.new_line(components=list(elems_not_in_lines))
        new_env.import_line(line=dummy_line, line_name='__DUMMY__')
        del new_env.lines['__DUMMY__'] # keep the elements but not the line

        env = new_env

    env.vars.default_to_zero = False

    return env