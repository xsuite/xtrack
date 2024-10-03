from typing import Dict

import numpy as np

import xtrack as xt
from xtrack.mad_parser.parse import ElementType, LineType, MadxParser, VarType


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


class MadxLoader:
    def __init__(self):
        self.env = xt.Environment()
        self._init_env()

    def _init_env(self):
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

    def load_file(self, file):
        parser = MadxParser()
        parsed_dict = parser.parse_file(file)

        self._parse_vars(parsed_dict["vars"])
        self._parse_elements(parsed_dict["elements"])
        self._parse_lines(parsed_dict["lines"])
        self._parse_parameters(parsed_dict["parameters"])

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

    def _parse_lines(self, lines: Dict[str, LineType]):
        for name, line_params in lines.items():
            params = line_params.copy()
            assert params.pop('parent') == 'sequence'
            builder = self.env.new_builder(name=name)
            self._parse_components(builder, params.pop('elements'))
            builder.build()

    def _parse_parameters(self, parameters: Dict[str, Dict[str, str]]):
        for element, el_params in parameters.items():
            params, extras = get_params(el_params, parent=element)
            self.env.set(element, **params, extra=extras)

    def _parse_components(self, builder, elements: Dict[str, LineType]):
        for name, element in elements.items():
            params = element.copy()
            parent = params.pop('parent')
            assert parent != 'sequence'
            params, extras = get_params(params, parent=parent)
            builder.new(name, parent, **params, extra=extras)
