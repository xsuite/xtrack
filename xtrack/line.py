# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

import math
import json
import logging
from collections import defaultdict
from weakref import WeakSet
from collections.abc import Iterable

from contextlib import contextmanager
from copy import deepcopy
from pprint import pformat
from typing import List, Literal, Optional, Dict
from pathlib import Path

import numpy as np
from scipy.constants import c as clight

from xdeps.refs import is_ref
from . import json as json_utils

import xobjects as xo
import xtrack as xt
import xdeps as xd
from .beam_elements.magnets import (
    MagnetEdge, _MODEL_TO_INDEX_CURVED,
    _EDGE_MODEL_TO_INDEX,
)
from .progress_indicator import progress
from .slicing import Custom, Slicer, Strategy
from .mad_writer import to_madx_sequence
from .madng_interface import (build_madng_model, discard_madng_model,
                              regen_madng_model, _tw_ng, line_to_madng,
                              _survey_ng)

from .survey import survey_from_line
from xtrack.twiss import (compute_one_turn_matrix_finite_differences,
                          find_closed_orbit_line, twiss_line,
                          compute_T_matrix_line,
                          get_non_linear_chromaticity,
                          DEFAULT_MATRIX_STABILITY_TOL,
                          DEFAULT_MATRIX_RESPONSIVENESS_TOL)
from xtrack.aperture_meas import measure_aperture
from .match import match_line, closed_orbit_correction, match_knob_line, Action
from .tapering import compensate_radiation_energy_loss
from .mad_loader import MadLoader
from .beam_elements import element_classes
from . import beam_elements
from .beam_elements import Drift, BeamElement, Marker, Multipole
from .beam_elements.slice_base import ID_RADIATION_FROM_PARENT
from .footprint import Footprint, _footprint_with_linear_rescale
from .internal_record import (start_internal_logging_for_elements_of_type,
                              stop_internal_logging_for_elements_of_type,
                              stop_internal_logging)
from .trajectory_correction import TrajectoryCorrection

from .general import _print

log = logging.getLogger(__name__)


_ALLOWED_ELEMENT_TYPES_IN_NEW = [xt.Drift, xt.Bend, xt.Quadrupole, xt.Sextupole,
                                 xt.Octupole, xt.Cavity, xt.Multipole,
                                 xt.UniformSolenoid, xt.Solenoid, xt.VariableSolenoid,
                                 xt.Marker, xt.Replica, xt.XYShift, xt.XRotation,
                                 xt.YRotation, xt.SRotation, xt.ZetaShift,
                                 xt.LimitRacetrack, xt.LimitRectEllipse,
                                 xt.LimitRect, xt.LimitEllipse,
                                 xt.LimitPolygon, xt.RFMultipole, xt.RBend,
                                 xt.Magnet]


_ALLOWED_ELEMENT_TYPES_DICT = {
    cc.__name__: cc for cc in _ALLOWED_ELEMENT_TYPES_IN_NEW}

_STR_ALLOWED_ELEMENT_TYPES_IN_NEW = ', '.join([tt.__name__ for tt in _ALLOWED_ELEMENT_TYPES_IN_NEW])


def find_index_repeated(item, lst,count=0):
    res=[ii for ii, nn in enumerate(lst) if nn == item]
    print(item)
    if count>=len(res):
        raise ValueError(f'Item {item} not found')
    return res[count]

def find_index_repeated2(item, lst,count=0):
    cc=0
    for ii, nn in enumerate(lst):
        if nn == item:
            if cc==count:
                return ii
            cc+=1
    raise ValueError(f'Item {item} not found')

class Line:

    """
    Beam line object. `Line.element_names` contains the ordered list of beam
    elements, `Line.element_dict` is a dictionary associating to each name the
    corresponding beam element object.
    """

    _element_dict = None
    config = None

    def __init__(self, elements=(), element_names=None, particle_ref=None,
                 energy_program=None, env=None):
        """
        Parameters
        ----------
        elements : dict or list of beam elements
            If a dictionary, it must be a dictionary associating to each name
            the corresponding beam element object. If a list, it must be a list
            of beam elements having the same length as the provided `element_names`.
        element_names : list of str
            Ordered list of beam element names. If not provided, `elements` must
            be a list, the names are automatically generated.
        particle_ref : xpart.Particles
            Reference particle providing rest mass, charge and reference enegy
            used for building particles distributions, computing twiss parameters
            and matching.
        energy_program: EnergyProgram
            (optional) Energy program used to update the reference energy during the tracking.
        env : Environment
            Environment object to which the line belongs. If not provided, a new
            environment is created.

        """

        self.config = xt.tracker.TrackerConfig()
        self.config.XTRACK_MULTIPOLE_NO_SYNRAD = True
        self.config.XFIELDS_BB3D_NO_BEAMSTR = True
        self.config.XFIELDS_BB3D_NO_BHABHA = True
        self.config.XTRACK_GLOBAL_XY_LIMIT = 1.0

        # Config parameters not exposed to C code
        # (accessed by user through properties)
        self._extra_config = {}
        self._extra_config['skip_end_turn_actions'] = False
        self._extra_config['reset_s_at_end_turn'] = True
        self._extra_config['matrix_responsiveness_tol'] = DEFAULT_MATRIX_RESPONSIVENESS_TOL
        self._extra_config['matrix_stability_tol'] = DEFAULT_MATRIX_STABILITY_TOL
        self._extra_config['dt_update_time_dependent_vars'] = 0.
        self._extra_config['_t_last_update_time_dependent_vars'] = None
        self._extra_config['_radiation_model'] = None
        self._extra_config['_beamstrahlung_model'] = None
        self._extra_config['_bhabha_model'] = None
        self._extra_config['_spin_model'] = None
        self._extra_config['_needs_rng'] = False
        self._extra_config['enable_time_dependent_vars'] = False
        self._extra_config['twiss_default'] = {}
        self._extra_config['steering_monitors_x'] = None
        self._extra_config['steering_monitors_y'] = None
        self._extra_config['steering_correctors_x'] = None
        self._extra_config['steering_correctors_y'] = None
        self._extra_config['corrector_limits_x'] = None
        self._extra_config['corrector_limits_y'] = None

        if env is None:
            env = xt.Environment()

        self.env = env
        self._element_dict = env.element_dict # Avoid copying (the property setter would do that)
        self._var_management = env._var_management
        self.env._lines_weakrefs.add(self)

        if particle_ref is None:
            particle_ref = env.particle_ref

        if isinstance(elements, dict):
            element_dict = elements
            if element_names is None:
                element_names = list(element_dict.keys())
        else:
            if element_names is None:
                element_names = [f"e{ii}" for ii in range(len(elements))]

            assert len(element_names) == len(elements), (
                "`elements` and `element_names` should have the same length"
            )
            element_dict = dict(zip(element_names, elements))

        self.element_dict.update(element_dict)
        self.element_names = list(element_names).copy()

        self.particle_ref = particle_ref

        if energy_program is not None:
            self.energy_program = energy_program # setter will take care of completing

        self.tracker = None

        self.metadata = {}

        self._line_before_slicing_cache = None
        self._element_names_before_slicing = None
        self.ref = xt.environment.EnvRef(self)

    @classmethod
    def from_dict(cls, dct, _context=None, _buffer=None, classes=(),
                  env=None, verbose=True):

        """
        Create a Line object from a dictionary.

        Parameters
        ----------
        dct : dict
            Dictionary containing the line data.
        _context : xobjects.Context, optional
            Context used for allocating the element data. If not provided the
            default xobjects context is used.
        _buffer : xobjects.Buffer, optional
            Buffer used for allocating the element data. If not provided, a new
            buffer is created.
        classes : list of classes, optional
            List of classes to be used for deserializing the elements. If not
            provided, the default classes are used.

        Returns
        -------
        line : Line
            Line object.

        """

        class_dict = mk_class_namespace(classes)

        _buffer = xo.get_a_buffer(context=_context, buffer=_buffer,size=8)

        if env is not None:
            elements = env.element_dict
        elif isinstance(dct['elements'], dict):
            elements = {}
            for ii, (kk, ee) in enumerate(
                    progress(dct['elements'].items(), desc='Loading line from dict')):
                elements[kk] = _deserialize_element(ee, class_dict, _buffer)
        elif isinstance(dct['elements'], list):
            elements = []
            for ii, ee in enumerate(
                    progress(dct['elements'], desc='Loading line from dict')):
                elements.append(_deserialize_element(ee, class_dict, _buffer))
        else:
            raise ValueError('Field `elements` must be a dict or a list')

        element_names = dct.get('element_names', [])
        self = cls(elements=elements, element_names=element_names)

        if 'particle_ref' in dct.keys():
            self.particle_ref = xt.Particles.from_dict(dct['particle_ref'],
                                    _context=_buffer.context)

        if env is not None:
            self.env = env
            self._var_management = env._var_management
            self._element_dict = env.element_dict # __init__ makes a copy of the dict
        elif '_var_manager' in dct.keys():
            # reinit env and var management
            self.env = None
            self._var_management = None
            self._init_var_management(dct=dct)

        if 'config' in dct.keys():
            self.config.clear()
            self.config.data.update(dct['config'])

        if '_extra_config' in dct.keys():
            self._extra_config.update(dct['_extra_config'])

        if 'metadata' in dct.keys():
            self.metadata = dct['metadata']

        self._element_names_before_slicing = dct.get(
            '_element_names_before_slicing', None)

        if ('energy_program' in self.element_dict
             and self.element_dict['energy_program'] is not None):
            self.energy_program.line = self

        if verbose:
            _print('Done loading line from dict.           ')

        return self

    @classmethod
    def from_json(cls, file, **kwargs):

        """Constructs a line from a json file.

        Parameters
        ----------
        file : str or file-like object
            Path to the json file or file-like object.
            If filename ends with '.gz' file is decompressed.
        **kwargs : dict
            Additional keyword arguments passed to `Line.from_dict`.

        Returns
        -------
        line : Line
            Line object.

        """

        dct = json_utils.load(file)

        if 'line' in dct.keys():
            dct_line = dct['line']
        else:
            dct_line = dct

        return cls.from_dict(dct_line, **kwargs)

    @classmethod
    def from_sequence(cls, nodes=None, length=None, elements=None,
                      sequences=None, copy_elements=False,
                      naming_scheme='{}{}', auto_reorder=False,
                      refer: Literal['entry', 'centre', 'exit'] = 'entry',
                      **kwargs):

        """

        Constructs a line from a sequence definition, inserting drift spaces
        as needed.

        Parameters
        ----------
        nodes : list of Node
            Sequence definition.
        length : float
            Total length (in m) of line. Determines drift behind last element.
        elements : dict
            Dictionary with named elements, which can be refered to in the
            sequence definion by name.
        sequences : dict
            Dictionary with named sub-sequences, which can be refered to in the
            sequence definion by name.
        copy_elements : bool, optional
            Whether to make copies of elements or not. By default, named elements
            are re-used which is memory efficient but does not allow to change
            parameters individually.
        naming_scheme : str, optional
            Naming scheme to name sub-sequences. A format string accepting two
            names to be joined.
        auto_reorder : bool, optional
            If false (default), nodes must be defined in order of increasing `s`
            coordinate, otherwise an exception is thrown. If true, nodes can be
            defined in any order and are re-ordered as necessary. Useful to
            place additional elements inside of sub-sequences.
        refer : str, optional
            Specifies where in the node the s coordinate refers to. Can be
            'entry', 'centre' or 'exit'. By default given s specifies the
            entry point of the element. If 'centre' is given, the s coordinate
            marks the centre of the element. If 'exit' is given, the s coordinate
            marks the exit point of the element.
        **kwargs : dict
            Arguments passed to constructor of the line

        Returns
        -------
        line : Line
            Line object.

        Examples
        --------

        .. code-block:: python
            from xtrack import Line, Node, Multipole
            elements = {
                    'quad': Multipole(length=0.3, knl=[0, +0.50]),
                    'bend': Multipole(length=0.5, knl=[np.pi / 12], hxl=[np.pi / 12]),
                }
            sequences = {
                    'arc': [Node(1, 'quad'), Node(5, 'bend')],
                }
            monitor = ParticlesMonitor(...)

            line = Line.from_sequence([
                    # direct element definition
                    Node(3, xt.Multipole(...)),
                    Node(7, xt.Multipole(...), name='quad1'),
                    Node(1, xt.Multipole(...), name='bend1', from_='quad1'),
                    ...
                    # using pre-defined elements by name
                    Node(13, 'quad'),
                    Node(14, 'quad', name='quad3'),
                    Node(2, 'bend', from_='quad3', name='bend2'),
                    ....
                    # using nested sequences
                    Node(5, 'arc', name='section_1'),
                    Node(3, monitor, from_='section_1'),
                ], length = 5, elements=elements, sequences=sequences)

        """

        # flatten the sequence
        nodes = flatten_sequence(nodes, elements=elements, sequences=sequences,
            copy_elements=copy_elements, naming_scheme=naming_scheme)
        if auto_reorder:
            nodes = sorted(nodes, key=lambda node: node.s)

        # add drifts
        element_objects = []
        element_names = []
        drifts = {}
        last_s = 0
        for node in nodes:
            if _is_thick(node.what, None):
                node_length = node.what.length
                if refer == 'entry':
                    offset = 0
                elif refer == 'centre':
                    offset = -node_length / 2
                elif refer == 'exit':
                    offset = -node_length
            else:
                node_length = 0
                offset = 0

            node_s = node.s + offset

            if node_s < last_s:
                raise ValueError(
                    f'Negative drift space from {last_s} to {node_s} '
                    f'({node.name} {refer}). Fix or set auto_reorder=True')

            # insert drift as needed (re-use if possible)
            if node_s > last_s:
                ds = node_s - last_s
                if ds not in drifts:
                    drifts[ds] = Drift(length=ds)
                element_objects.append(drifts[ds])
                element_names.append(_next_name('drift', element_names, naming_scheme))

            # insert element
            element_objects.append(node.what)
            element_names.append(node.name)
            last_s = node_s + node_length

        # add last drift
        if length < last_s:
            raise ValueError(f'Last element {node.name} at s={last_s} is outside length {length}')
        element_objects.append(Drift(length=length - last_s))
        element_names.append(_next_name('drift', element_names, naming_scheme))

        return cls(elements=element_objects, element_names=element_names, **kwargs)

    @classmethod
    def from_sixinput(cls, sixinput, classes=()):
        """
        Build a Line from a Sixtrack input object. N.B. This is a convenience
        function that calls sixinput.generate_xtrack_line(). It is used only for
        testing and will be removed in future versions.

        Parameters
        ----------

        sixinput : SixInput
            Sixtrack input object
        classes : tuple
            Tuple of classes to be used for the elements. If empty, the default
            classes are used.

        Returns
        -------
        line : Line
            Line object.

        """

        log.warning("\n"
            "WARNING: xtrack.Line.from_sixinput(sixinput) will be removed in future versions.\n"
            "Please use sixinput.generate_xtrack_line()\n")
        line = sixinput.generate_xtrack_line(classes=classes)
        return line

    @classmethod
    def from_madx_sequence(
        cls,
        sequence,
        deferred_expressions=False,
        install_apertures=False,
        apply_madx_errors=None,
        enable_field_errors=None,
        enable_align_errors=None,
        skip_markers=False,
        merge_drifts=False,
        merge_multipoles=False,
        expressions_for_element_types=None,
        replace_in_expr=None,
        classes=(),
        ignored_madtypes=(),
        allow_thick=None,
        name_prefix=None,
        enable_layout_data=False,
        enable_thick_kickers=False,
    ):

        """
        Build a line from a MAD-X sequence.

        Parameters
        ----------
        sequence : madx.Sequence
            MAD-X sequence object or name of the sequence
        deferred_expressions : bool, optional
            If true, deferred expressions from MAD-X are imported and can be
            accessed in `Line.vars` and `Line.element_refs`.
        install_apertures : bool, optional
            If true, aperture information is installed in the line.
        apply_madx_errors : bool, optional
            If true, errors are applied to the line.
        enable_field_errors : bool, optional
            If true, field errors are imported.
        enable_align_errors : bool, optional
            If true, alignment errors are imported.
        skip_markers : bool, optional
            If true, markers are skipped.
        merge_drifts : bool, optional
            If true, consecutive drifts are merged.
        merge_multipoles : bool, optional
            If true,consecutive multipoles are merged.
        expressions_for_element_types : list, optional
            List of element types for which expressions are imported.
        replace_in_expr : dict, optional
            Dictionary of replacements to be applied to expressions before they
            are imported.
        classes : tuple, optional
            Tuple of classes to be used for the elements. If empty, the default
            classes are used.
        ignored_madtypes : tuple, optional
            Tuple of MAD-X element types to be ignored.
        allow_thick : bool, optional
            If true, thick elements are allowed. Otherwise, an error is raised
            if a thick element is encountered.
        enable_layout_data: bool, optional
            If true, the layout data is imported.

        Returns
        -------
        line : Line
            Line object.

        """

        class_namespace = mk_class_namespace(classes)

        loader = MadLoader(
            sequence,
            classes=class_namespace,
            ignore_madtypes=ignored_madtypes,
            enable_errors=apply_madx_errors,
            enable_field_errors=enable_field_errors,
            enable_align_errors=enable_align_errors,
            enable_apertures=install_apertures,
            enable_expressions=deferred_expressions,
            skip_markers=skip_markers,
            merge_drifts=merge_drifts,
            merge_multipoles=merge_multipoles,
            expressions_for_element_types=expressions_for_element_types,
            error_table=None,  # not implemented yet
            replace_in_expr=replace_in_expr,
            allow_thick=allow_thick,
            name_prefix=name_prefix,
            enable_layout_data=enable_layout_data,
            enable_thick_kickers=enable_thick_kickers,
        )
        line = loader.make_line()
        return line

    def to_dict(self, include_var_management=True, include_element_dict=True):

        '''
        Returns a dictionary representation of the line.

        Parameters
        ----------
        include_var_management : bool, optional
            If True (default) the dictionary will contain the information
            needed to restore the line with deferred expressions.

        Returns
        -------
        out : dict
            Dictionary representation of the line.
        '''

        out = {}
        if include_element_dict:
            out["elements"] = {k: el.to_dict() for k, el in self.element_dict.items()}
        out["element_names"] = self.element_names[:]
        out['config'] = self.config.data.copy()
        out['_extra_config'] = self._extra_config.copy()

        if self._element_names_before_slicing is not None:
            out['_element_names_before_slicing'] = self._element_names_before_slicing

        if self.particle_ref is not None:
            out['particle_ref'] = self.particle_ref.to_dict()
        if self._var_management is not None and include_var_management:
            if hasattr(self, '_in_multiline') and self._in_multiline is not None:
                raise ValueError('The line is part ot a MultiLine object. '
                    'To save without expressions please use '
                    '`line.to_dict(include_var_management=False)`.\n'
                    'To save also the deferred expressions please save the '
                    'entire multiline.\n ')

            out.update(self._var_management_to_dict())

        out["metadata"] = deepcopy(self.metadata)

        return out

    def to_madx_sequence(self, sequence_name, mode='sequence'):
        '''
        Return a MAD-X sequence corresponding to the line.

        Parameters
        ----------
        sequence_name : str
            Name of the sequence.

        Returns
        -------
        madx_sequence : str
            MAD-X sequence.
        '''
        return to_madx_sequence(self, sequence_name, mode=mode)

    def to_madng(self, sequence_name='seq', temp_fname=None, keep_files=False,
                 **kwargs):

        '''
        Build a MAD NG instance from present state of the line.

        Parameters
        ----------
        sequence_name : str
            Name of the sequence.
        temp_fname : str
            Name of the temporary file to be used for the MAD NG instance.

        Returns
        -------
        mng : MAD
            MAD NG instance.
        '''

        return line_to_madng(self, sequence_name=sequence_name,
                             temp_fname=temp_fname, keep_files=keep_files,
                             **kwargs)


    build_madng_model = build_madng_model
    discard_madng_model = discard_madng_model
    regen_madng_model = regen_madng_model
    madng_twiss = _tw_ng
    madng_survey = _survey_ng

    def __repr__(self):
        if hasattr(self, '_name'):
            name = self._name
        else:
            name = ''
        return f'<{self.__class__.__name__} {name} at {id(self)}>'

    def __getstate__(self):
        out = self.__dict__.copy()
        return out

    def __setstate__(self, state):
        self.__dict__.update(state)


    def to_json(self, file, indent=1, **kwargs):
        '''Save the line to a json file.

        Parameters
        ----------
        file: str or file-like object
            The file to save to. If a string is provided, a file is opened and
            closed. If a file-like object is provided, it is used directly.
        **kwargs:
            Additional keyword arguments are passed to the `Line.to_dict` method.

        '''

        json_utils.dump(self.to_dict(**kwargs), file, indent=indent)

    def _to_table_dict(self):

        elements = list(self.elements)
        s_elements = np.array(list(self.get_s_elements()) + [self.get_length()])
        length_elements = np.diff(s_elements, append=s_elements[-1]) # only think elements have length here
        s_start = s_elements
        s_end = s_elements + length_elements
        s_center = s_start + 0.5 * length_elements

        isthick = []
        iscollective = []
        element_types = []
        isreplica = []
        parent_name = []
        for ee in elements:
            ee_pname = None
            if isinstance(ee, xt.Replica):
                ee_pname = ee.parent_name
                ee = ee.resolve(self)
                isreplica.append(True)
            else:
                isreplica.append(False)
                if hasattr(ee, 'parent_name'):
                    ee_pname = ee.parent_name
            isthick.append(_is_thick(ee, self))
            iscollective.append(_is_collective(ee, self))
            element_types.append(ee.__class__.__name__)
            parent_name.append(ee_pname)
        isthick = np.array(isthick + [False])
        iscollective = np.array(iscollective + [False])
        isreplica = np.array(isreplica + [False])
        element_types = np.array(element_types + [''])
        parent_name = np.array(parent_name + [None])

        elements += [None]

        out = {
            's': s_elements,
            'element_type': element_types,
            'name': list(self.element_names) + ['_end_point'],
            'isthick': isthick,
            'isreplica': isreplica,
            'parent_name': parent_name,
            'iscollective': iscollective,
            'element': elements,
            's_start': s_start,
            's_center': s_center,
            's_end': s_end,
        }

        return out

    def to_pandas(self):
        '''
        Return a pandas DataFrame with the elements of the line.

        Returns
        -------
        line_df : pandas.DataFrame
            DataFrame with the elements of the line.
        '''
        import pandas as pd

        elements_df = pd.DataFrame(self._to_table_dict())
        return elements_df

    def get_table(self, attr=False):

        data = self._to_table_dict()
        data.pop('element')

        if attr:
            for kk in self.attr.keys():
                this_attr = self.attr[kk]
                if hasattr(this_attr, 'get'):
                    this_attr = this_attr.get() # bring to cpu
                # Add zero at the end (there is _end_point)
                data[kk] = np.concatenate((this_attr, [this_attr[-1]*0]))

        for kk in data.keys():
            data[kk] = np.array(data[kk])

        names_table = xd.Table(data={'name': data['name']})
        names_unique = names_table.cols.get_index_unique()
        data['env_name'] = data['name']
        data['name'] = names_unique
        out = xd.Table(data=data, sep_count='::::')
        return out

    def get_strengths(self, reverse=None):

        if reverse is None:
            reverse = self.twiss_default.get('reverse', False)

        out = {}
        out['name'] = np.array(list(self.element_names) + ['_end_point'])
        for kk in (xt.twiss.NORMAL_STRENGTHS_FROM_ATTR
                 + xt.twiss.SKEW_STRENGTHS_FROM_ATTR
                 + xt.twiss.OTHER_FIELDS_FROM_ATTR):
            this_attr = self.attr[kk]
            if hasattr(this_attr, 'get'):
                this_attr = this_attr.get() # bring to cpu
            # Add zero at the end (there is _end_point)
            out[kk] = np.concatenate((this_attr, [this_attr[-1]*0]))

        if reverse:
            for kk in out:
                # Change order
                out[kk][:-1] = out[kk][:-1][::-1]

        tab = xt.Table(out)
        if reverse:
            xt.twiss._reverse_strengths(tab._data) # Change signs

        tab._data['reference_frame'] = {
            True: 'reverse', False: 'proper'}[reverse]
        return tab

    def get_aperture_table(self, dx=1e-3, dy=1e-3, x_range=(-0.1, 0.1),
                           y_range=(-0.1, 0.1)):
        '''
        Return a table with the horizontal and vertical aperture estimated at all
        elements of the line.
        The aperture is estimated by tracking a particle through the line and
        measuring the maximum and minumum horizontal and vertical position
        at which particles survive. For elements at which no lost particles are
        detected, the aperture is estimated by interpolating the values
        of the neighbouring elements.

        Parameters
        ----------
        dx : float, optional
            Required horizontal resolution (in m) for the aperture measurement.
            Default is 1e-3.
        dy : float, optional
            Required vertical resolution (in m) for the aperture measurement.
            Default is 1e-3.
        x_range : tuple, optional
            Horizontal range (in m) for the aperture measurement.
            Default is (-0.1, 0.1).
        y_range : tuple, optional
            Vertical range (in m) for the aperture measurement.
            Default is (-0.1, 0.1).

        Returns
        -------
        aperture_table : xtrack.Table
            Table with the horizontal and vertical aperture at all elements
            of the line.
        '''

        return xt.aperture_meas.measure_aperture(self,
            dx=1e-3, dy=1e-3, x_range=(-0.1, 0.1), y_range=(-0.1, 0.1))

    def copy(self, shallow=False, _context=None, _buffer=None):
        '''
        Return a copy of the line.

        Parameters
        ----------
        shallow : bool, optional
            If False (default), a deep copy is returned.
            If True, a shallow copy is returned, i.e. the line is plced in the
            same environment and shares variables and elements with the original.
        _context: xobjects.Context
            xobjects context to be used for the copy
        _buffer: xobjects.Buffer
            xobjects buffer to be used for the copy

        Returns
        -------
        line_copy : Line
            Copy of the line.
        '''

        if shallow==True:
            assert _context is None and _buffer is None, (
                'Shallow copy with _context or _buffer is not supported')
            out = self.select()
        else:
            elements = {nn: ee.copy(_context=_context, _buffer=_buffer)
                                        for nn, ee in self.element_dict.items()}
            element_names = [nn for nn in self.element_names]
            out = self.__class__(elements=elements, element_names=element_names)

            if self._var_management is not None:
                # reinit env and var management
                out.env = None
                out._var_management = None
                out._init_var_management(dct=self._var_management_to_dict())
                out._env_if_needed()

        if self.particle_ref is not None:
            out.particle_ref = self.particle_ref.copy(
                                        _context=_context, _buffer=_buffer)

        out.config.clear()
        out.config.update(self.config.copy())
        out._extra_config.update(self._extra_config.copy())
        out.metadata.clear()
        out.metadata.update(self.metadata)

        if out.energy_program is not None:
            out.energy_program.line = out

        return out

    def build_tracker(
            self,
            _context=None,
            _buffer=None,
            compile=True,
            io_buffer=None,
            use_prebuilt_kernels=True,
            enable_pipeline_hold=False,
            **kwargs):

        """
        Build the tracker associated to the line. This freezes the line (elements
        cannot be inserted or removed anymore). Use `discard_tracker` to unfreeze
        the line if needed.

        Parameters
        ----------
        _context: xobjects.Context, optional
            xobjects context to which the line data is moved and on which the
            tracking is performed. If not provided, the xobjects default context
            is used.
        _buffer: xobjects.Buffer
            xobjects buffer to which the line data is moved. If not provided,
            the _buffer is creted from the _context.
        compile: bool, optional
            If True (default) the tracker is compiled. If False, the tracker
            is not compiled until the first usage.
        io_buffer: xobjects.Buffer, optional
            xobjects buffer to be used for the I/O. If not provided, a new
            buffer is created.
        use_prebuilt_kernels: bool, optional
            If True (default) the prebuilt kernels are used if available.
            If False, the kernels are always compiled.
        enable_pipeline_hold: bool, optional
            If True, the pipeline hold mechanism is enabled.

        """

        if self.tracker is not None:
            _print('The line already has an associated tracker')
            return self.tracker

        if (len(self.element_names) == 0 and hasattr(self, 'builder')
            and self.builder is not None):
            self.rebuild()

        self.tracker = xt.Tracker(
                                line=self,
                                _context=_context,
                                _buffer=_buffer,
                                compile=compile,
                                io_buffer=io_buffer,
                                use_prebuilt_kernels=use_prebuilt_kernels,
                                enable_pipeline_hold=enable_pipeline_hold,
                                **kwargs)

        if hasattr(self, 'env') and self.env is not None:
            self.env._ensure_tracker_consistency(buffer=self._buffer)

        return self.tracker

    def rebuild(self):
        if not hasattr(self, 'builder') or self.builder is None:
            raise ValueError('The line does not have a builder')

        temp = self.builder.build()
        self.element_names = temp.element_names

    @property
    def attr(self):

        if not self._has_valid_tracker():
            self.build_tracker()

        if ('attr' not in self.tracker._tracker_data_base.cache.keys()
                or self.tracker._tracker_data_base.cache['attr'] is None):
            self.tracker._tracker_data_base.cache['attr'] = self._get_attr_cache()

        return self.tracker._tracker_data_base.cache['attr']

    @property
    def particle_ref(self):
        return self._particle_ref

    @particle_ref.setter
    def particle_ref(self, particle_ref):
        self._particle_ref = particle_ref
        if self.particle_ref is not None and self.particle_ref.t_sim == 0:
            self.particle_ref.t_sim = (
                self.get_length() / self.particle_ref._xobject.beta0[0] / clight)

    @property
    def scattering(self):
        if not hasattr(self, '_scattering') or self._scattering is None:
            try:
                from xcoll.line_tools import XcollScatteringAPI
                self._scattering = XcollScatteringAPI(line=self)
            except ImportError as error:
                raise ImportError("Please install Xcoll to use this feature.") from error

        return self._scattering

    @property
    def collimators(self):
        if not hasattr(self, '_collimators') or self._collimators is None:
            try:
                from xcoll.line_tools import XcollCollimatorAPI
                self._collimators = XcollCollimatorAPI(line=self)
            except ImportError as error:
                raise ImportError("Please install Xcoll to use this feature.") from error

        return self._collimators

    def _get_bucket(self):
        import xpart as xp
        return xp.longitudinal.get_bucket(self)

    def discard_tracker(self):

        """
        Discard the tracker associated to the line. This unfreezes the line
        (elements can be inserted or removed again).

        """
        if not isinstance(self._element_names, list):
            self._element_names = list(self._element_names)
        if hasattr(self, 'tracker') and self.tracker is not None:
            self.tracker._invalidate()
            self.tracker = None

    def track(
        self,
        particles,
        ele_start=0,
        ele_stop=None,     # defaults to full lattice
        num_elements=None, # defaults to full lattice
        num_turns=None,    # defaults to 1
        turn_by_turn_monitor=None,
        freeze_longitudinal=False,
        time=False,
        with_progress=False,
        **kwargs):

        """
        Track particles through the line.

        Parameters
        ----------
        particles: xpart.Particles
            The particles to track
        ele_start: int or str, optional
            The element to start tracking from (inclusive). If an integer is
            provided, it is interpreted as the index of the element in the line.
            If a string is provided, it is interpreted as the name of the element
            in the line.
        ele_stop: int or str, optional
            The element to stop tracking at (exclusive). If an integer is provided,
            it is interpreted as the index of the element in the line. If a string
            is provided, it is interpreted as the name of the element in the line.
        num_elements: int, optional
            The number of elements to track through. If `ele_stop` is not
            provided, this is the number of elements to track through from
            `ele_start`. If `ele_stop` is provided, `num_elements` should not
            be provided.
        num_turns: int, optional
            The number of turns to track through. Defaults to 1.
        backetrack: bool, optional
            If True, the particles are tracked backward from ele_stop to ele_start.
        turn_by_turn_monitor: bool, str or xtrack.ParticlesMonitor, optional
            If True, a turn-by-turn monitor is created. If a monitor is provided,
            it is used directly. If the string `ONE_TURN_EBE` is provided, the
            particles coordinates are recorded at each element (one turn).
            The recorded data can be retrieved in `line.record_last_track`.
        freeze_longitudinal: bool, optional
            If True, the longitudinal coordinates are frozen during tracking.
        time: bool, optional
            If True, the time taken for tracking is recorded and can be retrieved
            in `line.time_last_track`.
        with_progress: bool or int, optional
            If truthy, a progress bar is displayed during tracking. If an integer
            is provided, it is used as the number of turns between two updates
            of the progress bar. If True, 100 is taken by default. By default,
            equals to False and no progress bar is displayed.
        """

        if not self._has_valid_tracker():
            self.build_tracker()

        if hasattr(particles, '_needs_pipeline') and particles._needs_pipeline:
            if '_called_by_pipeline' not in kwargs or not kwargs['_called_by_pipeline']:
                all_kwargs = locals()
                multitracker = xt.PipelineMultiTracker(
                    branches=[xt.PipelineBranch(line=self, particles=particles)])
                all_kwargs.pop('self')
                all_kwargs.pop('particles')
                all_kwargs.pop('kwargs')
                all_kwargs.pop('_called_by_pipeline', None)
                return multitracker.track(**all_kwargs, **kwargs)

        if '_called_by_pipeline' in kwargs: # Used only above
            kwargs.pop('_called_by_pipeline')

        if not self._has_valid_tracker():
            self.build_tracker()

        return self.tracker._track(
            particles,
            ele_start=ele_start,
            ele_stop=ele_stop,
            num_elements=num_elements,
            num_turns=num_turns,
            turn_by_turn_monitor=turn_by_turn_monitor,
            freeze_longitudinal=freeze_longitudinal,
            time=time,
            with_progress=with_progress,
            **kwargs)

    def slice_thick_elements(self, slicing_strategies):
        """
        Slice thick elements in the line. Slicing is done in place.

        Parameters
        ----------
        slicing_strategies : list
            List of slicing Strategy objects. In case multiple strategies
            apply to the same element, the last one takes precedence)

        Examples
        --------

        .. code-block:: python

            line.slice_thick_elements(
                slicing_strategies=[
                    # Slicing with thin elements
                    xt.Strategy(slicing=xt.Teapot(1)), # (1) Default applied to all elements
                    xt.Strategy(slicing=xt.Uniform(2), element_type=xt.Bend), # (2) Selection by element type
                    xt.Strategy(slicing=xt.Teapot(3), element_type=xt.Quadrupole),  # (4) Selection by element type
                    xt.Strategy(slicing=xt.Teapot(4), name='mb1.*'), # (5) Selection by name pattern
                    # Slicing with thick elements
                    xt.Strategy(slicing=xt.Uniform(2, mode='thick'), name='mqf.*'), # (6) Selection by name pattern
                    # Do not slice (leave untouched)
                    xt.Strategy(slicing=None, name='mqd.1') # (7) Selection by name
            ])

        """

        self.build_tracker(compile=False) # ensure elements are in the same buffer
        self.discard_tracker()

        self._line_before_slicing_cache = None
        self._element_names_before_slicing = list(self.element_names).copy()

        slicer = Slicer(self, slicing_strategies)
        return slicer.slice_in_place()

    def build_particles(
        self,
        particle_ref=None,
        num_particles=None,
        x=None, px=None, y=None, py=None, zeta=None, delta=None, pzeta=None,
        x_norm=None, px_norm=None, y_norm=None, py_norm=None, zeta_norm=None, pzeta_norm=None,
        at_element=None, match_at_s=None,
        nemitt_x=None, nemitt_y=None,
        weight=None,
        particle_on_co=None,
        R_matrix=None,
        W_matrix=None,
        method=None,
        scale_with_transverse_norm_emitt=None,
        include_collective=True,
        _context=None, _buffer=None, _offset=None,
        _capacity=None,
        mode=None,
        **kwargs, # They are passed to the twiss
    ):

        """
        Create a Particles object from arrays containing physical or
        normalized coordinates.

        Parameters
        ----------

        particle_ref : Particle object
            Reference particle defining the reference quantities (mass0, q0, p0c,
            gamma0, etc.). Its coordinates (x, py, y, py, zeta, delta) are ignored
            unless `mode`='shift' is selected.
        num_particles : int
            Number of particles to be generated (used if provided coordinates are
            all scalar).
        x : float or array
            x coordinate of the particles in meters (default is 0).
        px : float or array
            px coordinate of the particles (default is 0).
        y : float or array
            y coordinate of the particles in meters (default is 0).
        py : float or array
            py coordinate of the particles (default is 0).
        zeta : float or array
            zeta coordinate of the particles in meters (default is 0).
        delta : float or array
            delta coordinate of the particles (default is 0).
        pzeta : float or array
            pzeta coordinate of the particles (default is 0).
        x_norm : float or array
            transverse normalized coordinate x (in sigmas) used in combination with
            the one turn matrix and with the transverse emittances provided
            in the argument `scale_with_transverse_norm_emitt` to generate x, px,
            y, py (x, px, y, py cannot be provided if x_norm, px_norm, y_norm,
            py_norm are provided).
        px_norm : float or array
            transverse normalized coordinate px (in sigmas) used in combination
            with the one turn matrix and with the transverse emittances (as above).
        y_norm : float or array
            transverse normalized coordinate y (in sigmas) used in combination
            with the one turn matrix and with the transverse emittances (as above).
        py_norm : float or array
            transverse normalized coordinate py (in sigmas) used in combination
            with the one turn matrix and with the transverse emittances (as above).
        zeta_norm : float or array
            longitudinal normalized coordinate zeta (in sigmas) used in combination
            with the one turn matrix.
        pzeta_norm : float or array
            longitudinal normalized coordinate pzeta (in sigmas) used in combination
            with the one turn matrix.
        nemitt_x : float
            Transverse normalized emittance in the `x` plane.
        nemitt_y : float
            Transverse normalized emittance in the `y` plane.
        at_element : str or int
            Location within the line at which particles are generated. It can be an
            index or an element name.
        match_at_s : float
            `s` location in meters within the line at which particles are generated. The value
            needs to be in the drift downstream of the element at `at_element`.
            The matched particles are backtracked to the element at `at_element`
            from which the tracking automatically starts when the generated
            particles are tracked.
        weight : float or array
            weights to be assigned to the particles.
        mode : str
            To be chosen between `set`,  `shift` and `normalized_transverse` (the
            default mode is `set`. `normalized_transverse` is used if any if any
            of `x_norm`, `px_norm`, `y_norm`, `py_norm` is provided):
                - `set`: reference quantities including mass0, q0, p0c, gamma0,
                    etc. are taken from the provided reference particle. Particles
                    coordinates are set according to the provided input x, px, y, py,
                    zeta, delta (zero is assumed as default for these variables).
                - `shift`: reference quantities including mass0, q0, p0c, gamma0,
                    etc. are taken from the provided reference particle. Particles
                    coordinates are set from the reference particles and shifted
                    according to the provided input x, px, y, py, zeta, delta (zero
                    is assumed as default for these variables).
                - `normalized_transverse`: reference quantities including mass0,
                    q0, p0c, gamma0, etc. are taken from the provided reference
                    particle. The longitudinal coordinates are set according to the
                    provided input `zeta`, `delta` (zero is assumed as default for
                    these variables). The transverse coordinates are set according
                    to the provided input `x_norm`, `px_norm`, `y_norm`, `py_norm`
                    (zero is assumed as default for these variables). The
                    transverse coordinates are normalized according to the
                    transverse emittance provided in `nemitt_x` and `nemitt_y`.
                    The transverse coordinates are then transformed into physical
                    space using the linearized one-turn matrix.
        _capacity : int
            Capacity of the arrays to be created. If not provided, the capacity
            is set to the number of particles.

        Returns
        -------
        particles : Particles object
            Particles object containing the generated particles.

        """

        if not self._has_valid_tracker():
            self.build_tracker()

        import xpart
        return xpart.build_particles(
            line=self,
            particle_ref=particle_ref,
            num_particles=num_particles,
            x=x, px=px, y=y, py=py, zeta=zeta, delta=delta, pzeta=pzeta,
            x_norm=x_norm, px_norm=px_norm, y_norm=y_norm, py_norm=py_norm,
            zeta_norm=zeta_norm, pzeta_norm=pzeta_norm,
            at_element=at_element, match_at_s=match_at_s,
            nemitt_x=nemitt_x, nemitt_y=nemitt_y,
            weight=weight,
            particle_on_co=particle_on_co,
            R_matrix=R_matrix,
            W_matrix=W_matrix,
            method=method,
            scale_with_transverse_norm_emitt=scale_with_transverse_norm_emitt,
            _context=_context, _buffer=_buffer, _offset=_offset,
            _capacity=_capacity,
            mode=mode,
            include_collective=include_collective,
            **kwargs)

    def twiss(self, particle_ref=None, method=None,
        particle_on_co=None, R_matrix=None, W_matrix=None,
        delta0=None, zeta0=None,
        r_sigma=None, nemitt_x=None, nemitt_y=None,
        delta_disp=None, delta_chrom=None, zeta_disp=None,
        co_guess=None, steps_r_matrix=None,
        co_search_settings=None, at_elements=None, at_s=None,
        continue_on_closed_orbit_error=None,
        freeze_longitudinal=None,
        freeze_energy=None,
        values_at_element_exit=None,
        radiation_method=None,
        eneloss_and_damping=None,
        radiation_integrals=None,
        start=None, end=None, init=None,
        num_turns=None,
        skip_global_quantities=None,
        matrix_responsiveness_tol=None,
        matrix_stability_tol=None,
        symplectify=None,
        reverse=None,
        use_full_inverse=None,
        strengths=None,
        hide_thin_groups=None,
        search_for_t_rev=None,
        num_turns_search_t_rev=None,
        only_twiss_init=None,
        only_markers=None,
        only_orbit=None,
        spin=None,
        polarization=None,
        compute_R_element_by_element=None,
        compute_lattice_functions=None,
        compute_chromatic_properties=None,
        coupling_edw_teng=False,
        init_at=None,
        x=None, px=None, y=None, py=None, zeta=None, delta=None,
        betx=None, alfx=None, bety=None, alfy=None, bets=None,
        dx=None, dpx=None, dy=None, dpy=None, dzeta=None,
        mux=None, muy=None, muzeta=None,
        ax_chrom=None, bx_chrom=None, ay_chrom=None, by_chrom=None,
        ddx=None, ddpx=None, ddy=None, ddpy=None,
        spin_x=None, spin_y=None, spin_z=None,
        zero_at=None,
        co_search_at=None,
        include_collective=None,
        _continue_if_lost=None,
        _keep_tracking_data=None,
        _keep_initial_particles=None,
        _initial_particles=None,
        _ebe_monitor=None,
        ele_start='__discontinued__',
        ele_stop='__discontinued__',
        ele_init='__discontinued__',
        twiss_init='__discontinued__'
        ):

        if not self._has_valid_tracker():
            self.build_tracker()

        tw_kwargs = locals().copy()

        for old, new in zip(['ele_start', 'ele_stop', 'ele_init', 'twiss_init'],
                            ['start', 'end', 'init_at', 'init']):
            if tw_kwargs[old] != '__discontinued__':
                raise ValueError(f'`{old}` is deprecated. Please use `{new}`.')
            tw_kwargs.pop(old)

        for kk, vv in self.twiss_default.items():
            if kk not in tw_kwargs.keys() or tw_kwargs[kk] is None:
                tw_kwargs[kk] = vv

        tw_kwargs.pop('self')
        return twiss_line(self, **tw_kwargs)

    twiss.__doc__ = twiss_line.__doc__

    def twiss4d(self, **kwargs):

        """
        Compute the 4D Twiss parameters. Equivalent to `twiss` with `method='4d'`.
        """

        kwargs['method'] = '4d'
        return self.twiss(**kwargs)

    def match(self, vary, targets, solve=True, assert_within_tol=True,
                  compensate_radiation_energy_loss=False,
                  solver_options={}, allow_twiss_failure=True,
                  restore_if_fail=True, verbose=None,
                  n_steps_max=20, default_tol=None,
                  solver=None, check_limits=True, **kwargs):
        '''
        Change a set of knobs in the beamline in order to match assigned targets.

        Parameters
        ----------
        vary : list of str or list of Vary objects
            List of knobs to be varied. Each knob can be a string or a Vary object
            including the knob name and the step used for computing the Jacobian
            for the optimization.
        targets : list of Target objects
            List of targets to be matched.
        solve : bool
            If True (default), the matching is performed immediately. If not an
            Optimize object is returnd, which can be used for advanced matching.
        assert_within_tol : bool
            If True (default), an exception is raised if the matching fails.
        compensate_radiation_energy_loss : bool
            If True, the radiation energy loss is compensated at each step of the
            matching.
        solver_options : dict
            Dictionary of options to be passed to the solver.
        allow_twiss_failure : bool
            If True (default), the matching continues if the twiss computation
            computation fails at some of the steps.
        restore_if_fail : bool
            If True (default), the beamline is restored to its initial state if
            the matching fails.
        verbose : bool
            If True, the matching steps are printed.
        n_steps_max : int
            Maximum number of steps for the matching before matching is stopped.
        default_tol : float
            Default tolerances used on the target. A dictionary can be provided
            associating a tolerance to each target name. The tolerance provided
            for `None` is used for all targets for which a tolerance is not
            otherwise provided. Example: `default_tol={'betx': 1e-4, None: 1e-6}`.
        solver : str
            Solver to be used for the matching.
        check_limits : bool
            If True (default), the limits of the knobs are checked before the
            optimization. If False, if the knobs are out of limits, the optimization
            knobs are set to the limits on the first iteration.
        **kwargs : dict
            Additional arguments to be passed to the twiss.

        Returns
        -------
        optimizer : xdeps.Optimize
            xdeps optimizer object used for the optimization.

        Examples
        --------

        .. code-block:: python

            # Match tunes and chromaticities to assigned values
            line.match(
                vary=[
                    xt.Vary('kqtf.b1', step=1e-8),
                    xt.Vary('kqtd.b1', step=1e-8),
                    xt.Vary('ksf.b1', step=1e-8),
                    xt.Vary('ksd.b1', step=1e-8),
                ],
                targets = [
                    xt.Target('qx', 62.315, tol=1e-4),
                    xt.Target('qy', 60.325, tol=1e-4),
                    xt.Target('dqx', 10.0, tol=0.05),
                    xt.Target('dqy', 12.0, tol=0.05)]
            )

        .. code-block:: python

            # Match a local orbit bump
            tw_before = line.twiss()

            line.match(
                start='mq.33l8.b1',
                end='mq.23l8.b1',
                init=tw_before.get_twiss_init(at_element='mq.33l8.b1'),
                vary=[
                    xt.Vary(name='acbv30.l8b1', step=1e-10),
                    xt.Vary(name='acbv28.l8b1', step=1e-10),
                    xt.Vary(name='acbv26.l8b1', step=1e-10),
                    xt.Vary(name='acbv24.l8b1', step=1e-10),
                ],
                targets=[
                    # I want the vertical orbit to be at 3 mm at mq.28l8.b1 with zero angle
                    xt.Target('y', at='mb.b28l8.b1', value=3e-3, tol=1e-4, scale=1),
                    xt.Target('py', at='mb.b28l8.b1', value=0, tol=1e-6, scale=1000),
                    # I want the bump to be closed
                    xt.Target('y', at='mq.23l8.b1', value=tw_before['y', 'mq.23l8.b1'],
                            tol=1e-6, scale=1),
                    xt.Target('py', at='mq.23l8.b1', value=tw_before['py', 'mq.23l8.b1'],
                            tol=1e-7, scale=1000),
                ]
            )

        '''

        if not self._has_valid_tracker():
            self.build_tracker()

        for old, new in zip(['ele_start', 'ele_stop', 'ele_init', 'twiss_init'],
                                ['start', 'end', 'init_at', 'init']):
                if old in kwargs.keys():
                    raise ValueError(f'`{old}` is deprecated. Please use `{new}`.')

        return match_line(self,
                        vary=vary, targets=targets, solve=solve,
                        assert_within_tol=assert_within_tol,
                        compensate_radiation_energy_loss=compensate_radiation_energy_loss,
                        solver_options=solver_options,
                        allow_twiss_failure=allow_twiss_failure,
                        restore_if_fail=restore_if_fail,
                        verbose=verbose, n_steps_max=n_steps_max,
                        default_tol=default_tol, solver=solver,
                        check_limits=check_limits, **kwargs)


    def match_knob(self, knob_name, vary, targets,
                   knob_value_start=0, knob_value_end=1,
                   **kwargs):

        '''
        Match a new knob in the beam line such that the specified targets are
        matched when the knob is set to the value `knob_value_end` and the
        state of the line before tha matching is recovered when the knob is
        set to the value `knob_value_start`.

        Parameters
        ----------
        knob_name : str
            Name of the knob to be matched.
        vary : list of str or list of Vary objects
            List of existing knobs to be varied.
        targets : list of Target objects
            List of targets to be matched.
        knob_value_start : float
            Value of the knob before the matching. Defaults to 0.
        knob_value_end : float
            Value of the knob after the matching. Defaults to 1.

        '''
        if not self._has_valid_tracker():
            self.build_tracker()

        opt = match_knob_line(self, vary=vary, targets=targets,
                        knob_name=knob_name, knob_value_start=knob_value_start,
                        knob_value_end=knob_value_end, **kwargs)

        return opt


    def survey(self,X0=0,Y0=0,Z0=0,theta0=0, phi0=0, psi0=0,
               element0=0, reverse=None):

        """
        Returns a survey of the beamline (based on MAD-X survey command).

        Parameters
        ----------
        X0 : float
            Initial X coordinate in meters.
        Y0 : float
            Initial Y coordinate in meters.
        Z0 : float
            Initial Z coordinate in meters.
        theta0 : float
            Initial theta coordinate in radians.
        phi0 : float
            Initial phi coordinate in radians.
        psi0 : float
            Initial psi coordinate in radians.
        element0 : int or str
            Element at which the given coordinates are defined.

        Returns
        -------
        survey : SurveyTable
            Survey table.
        """

        if not self._has_valid_tracker():
            self.build_tracker()

        if reverse is None:
            reverse = self.twiss_default.get('reverse', False)

        return survey_from_line(self, X0=X0, Y0=Y0, Z0=Z0, theta0=theta0,
                                   phi0=phi0, psi0=psi0, element0=element0,
                                   reverse=reverse)

    def correct_trajectory(self, run=True, n_iter='auto', start=None, end=None,
                 twiss_table=None, planes=None,
                 monitor_names_x=None, corrector_names_x=None,
                 monitor_names_y=None, corrector_names_y=None,
                 n_micado=None, n_singular_values=None, rcond=None,
                 monitor_alignment=None, corrector_limits_x=None,
                 corrector_limits_y=None):

        '''
        Correct the beam trajectory using linearized response matrix from optics
        table.

        Parameters
        ----------

        run : bool
            If True (default), the correction is performed immediately. If False,
            a TrajectoryCorrection object is returned, which can be used for
            advanced correction.
        n_iter : int
            Number of iterations for the correction. If 'auto' (default), the
            iterations are performed for as long as the correction is improving.
        start : str
            Start of the line range in which the correction is performed.
            If `start` is provided `end` must also be provided.
            If `start` is None, the correction is performed on the periodic
            solution (closed orbit).
        end : str
            End of the line range in which the correction is performed.
            If `end` is provided `start` must also be provided.
            If `start` is None, the correction is performed on the periodic
            solution (closed orbit).
        twiss_table : TwissTable
            Twiss table used to compute the response matrix for the correction.
            If None, the twiss table is computed from the line.
        planes : str
            Planes for which the correction is performed. It can be 'x', 'y' or
            'xy'. If None, the correction is performed for both planes.
        monitor_names_x : list of str
            List of elements used as monitors in the horizontal plane.
        corrector_names_x : list of str
            List of elements used as correctors in the horizontal plane. They
            must have `knl` and `ksl` attributes.
        monitor_names_y : list of str
            List of elements used as monitors in the vertical plane.
        corrector_names_y : list of str
            List of elements used as correctors in the vertical plane. They
            must have `knl` and `ksl` attributes.
        n_micado : int
            If `n_micado` is not None, the MICADO algorithm is used for the
            correction. In that case, the number of correctors to be used is
            given by `n_micado`.
        n_singular_values : int
            Number of singular values used for the correction.
        rcond : float
            Cutoff for small singular values (relative to the largest singular
            value). Singular values smaller than `rcond` are considered zero.
        corrector_limits_x : tuple of array-like or None
            Limits for the horizontal corrector strengths. If not None, it should be a tuple
            of two arrays (lower_limits, upper_limits) with the same length as
            the number of horizontal correctors. If None, no limits are applied.
        corrector_limits_y : tuple of array-like or None
            Limits for the vertical corrector strengths. If not None, it should be a tuple
            of two arrays (lower_limits, upper_limits) with the same length as
            the number of vertical correctors. If None, no limits are applied.

        Returns
        -------
        correction : TrajectoryCorrection
            Trajectory correction object.

        '''

        correction = TrajectoryCorrection(line=self,
                 start=start, end=end, twiss_table=twiss_table,
                 monitor_names_x=monitor_names_x,
                 corrector_names_x=corrector_names_x,
                 monitor_names_y=monitor_names_y,
                 corrector_names_y=corrector_names_y,
                 n_micado=n_micado, n_singular_values=n_singular_values,
                 rcond=rcond,
                 monitor_alignment=monitor_alignment,
                 corrector_limits_x=corrector_limits_x,
                 corrector_limits_y=corrector_limits_y)

        if run:
            correction.correct(planes=planes, n_iter=n_iter)

        return correction

    def _xmask_correct_closed_orbit(self, reference, correction_config,
                        solver=None, verbose=False, restore_if_fail=True):

        """
        Correct the closed orbit of the beamline through a set of local matches.

        Parameters
        ----------
        reference : Line
            Line on which the reference closed orbit is computed.
        correction_config : dict
            Dictionary containing the configuration for the closed orbit correction.
            The dictionary must have the structure shown in the example below.
        solver : str
            Solver to be used for the matching. Available solvers are "fsolve"
            and "bfgs".
        verbose : bool
            If True, the matching steps are printed.
        restore_if_fail : bool
            If True, the beamline is restored to its initial state if the matching
            fails.

        Examples
        --------

        .. code-block:: python

            correction_config = {
                'IR1 left': dict(
                    ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
                    start='e.ds.r8.b1',
                    end='e.ds.l1.b1',
                    vary=(
                        'corr_co_acbh14.l1b1',
                        'corr_co_acbh12.l1b1',
                        'corr_co_acbv15.l1b1',
                        'corr_co_acbv13.l1b1',
                        ),
                    targets=('e.ds.l1.b1',),
                ),
                'IR1 right': dict(
                    ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
                    start='s.ds.r1.b1',
                    end='s.ds.l2.b1',
                    vary=(
                        'corr_co_acbh13.r1b1',
                        'corr_co_acbh15.r1b1',
                        'corr_co_acbv12.r1b1',
                        'corr_co_acbv14.r1b1',
                        ),
                    targets=('s.ds.l2.b1',),
                ),
                ...
            }

            line.correct_closed_orbit(
                reference=line_reference,
                correction_config=correction_config)

        """


        closed_orbit_correction(self, reference, correction_config,
                                solver=solver, verbose=verbose,
                                restore_if_fail=restore_if_fail)

    def find_closed_orbit(self, co_guess=None, particle_ref=None,
                          co_search_settings={}, delta_zeta=0,
                          delta0=None, zeta0=None,
                          continue_on_closed_orbit_error=False,
                          freeze_longitudinal=False,
                          start=None, end=None,
                          num_turns=1,
                          co_search_at=None,
                          search_for_t_rev=False,
                          spin=None,
                          num_turns_search_t_rev=None,
                          symmetrize=False,
                          include_collective=False):

        """
        Find the closed orbit of the beamline.

        Parameters
        ----------
        co_guess : Particles or dict
            Particle used as first guess to compute the closed orbit. If None,
            the reference particle is used.
        particle_ref : Particle
            Particle used to compute the closed orbit. If None, the reference
            particle is used.
        co_search_settings : dict
            Dictionary containing the settings for the closed orbit search
            (passed as keyword arguments to the `scipy.fsolve` function)
        delta_zeta : float
            Initial delta_zeta coordinate.
        delta0 : float
            Initial delta coordinate.
        zeta0 : float
            Initial zeta coordinate in meters.
        continue_on_closed_orbit_error : bool
            If True, the closed orbit at the last step is returned even if
            the closed orbit search fails.
        freeze_longitudinal : bool
            If True, the longitudinal coordinates are frozen during the closed
            orbit search.
        start : int or str
            Optional. It can be provided to find the periodic solution for
            a portion of the beamline.
        end : int or str
            Optional. It can be provided to find the periodic solution for
            a portion of the beamline.
        num_turns : int
            Number of turns to be used for the closed orbit search.
        co_search_at : int or str
            Element at which the closed orbit search is performed. If None,
            the closed orbit search is performed at the start of the line.

        Returns
        -------
        particle_on_co : Particle
            Particle at the closed orbit.

        """

        if freeze_longitudinal:
            kwargs = locals().copy()
            kwargs.pop('self')
            kwargs.pop('freeze_longitudinal')
            with _freeze_longitudinal(self):
                return self.find_closed_orbit(**kwargs)

        self._check_valid_tracker()

        if particle_ref is None and co_guess is None:
            particle_ref = self.particle_ref

        if self.iscollective and not include_collective:
            log.warning(
                'The tracker has collective elements.\n'
                'In the twiss computation collective elements are'
                ' replaced by drifts')
            line = self._get_non_collective_line()
        else:
            line = self

        return find_closed_orbit_line(line, co_guess=co_guess,
                                 particle_ref=particle_ref, delta0=delta0, zeta0=zeta0,
                                 co_search_settings=co_search_settings, delta_zeta=delta_zeta,
                                 continue_on_closed_orbit_error=continue_on_closed_orbit_error,
                                 start=start, end=end, num_turns=num_turns,
                                 co_search_at=co_search_at,
                                 search_for_t_rev=search_for_t_rev,
                                 spin=spin,
                                 num_turns_search_t_rev=num_turns_search_t_rev,
                                 symmetrize=symmetrize)

    def compute_T_matrix(self, start=None, end=None,
                         particle_on_co=None, steps_t_matrix=None):

        """
        Compute the second order tensor of the beamline.

        Parameters
        ----------
        start : int or str
            Element at which the computation starts.
        end : int or str
            Element at which the computation stops.
        particle_on_co : Particle
            Particle at the closed orbit (optional).
        steps_r_matrix : int
            Finite difference step for computing the second order tensor.

        Returns
        -------
        T_matrix : ndarray
            Second order tensor of the beamline.

        """

        self._check_valid_tracker()

        return compute_T_matrix_line(self, start=start, end=end,
                                particle_on_co=particle_on_co,
                                steps_t_matrix=steps_t_matrix)

    def get_footprint(self, nemitt_x=None, nemitt_y=None, n_turns=256, n_fft=2**18,
            mode='polar', r_range=None, theta_range=None, n_r=None, n_theta=None,
            x_norm_range=None, y_norm_range=None, n_x_norm=None, n_y_norm=None,
            linear_rescale_on_knobs=None,
            freeze_longitudinal=None, delta0=None, zeta0=None,
            keep_fft=True, keep_tracking_data=False):

        '''
        Compute the tune footprint for a beam with given emittences using tracking.

        Parameters
        ----------

        nemitt_x : float
            Normalized emittance in the x-plane.
        nemitt_y : float
            Normalized emittance in the y-plane.
        n_turns : int
            Number of turns for tracking.
        n_fft : int
            Number of points for FFT (tracking data is zero-padded to this length).
        mode : str
            Mode for computing footprint. Options are 'polar' and 'uniform_action_grid'.
            In 'polar' mode, the footprint is computed on a polar grid with
            r_range and theta_range specifying the range of r and theta values (
            polar coordinates in the x_norm, y_norm plane).
            In 'uniform_action_grid' mode, the footprint is computed on a uniform
            grid in the action space (Jx, Jy).
        r_range : tuple of floats
            Range of r values for footprint in polar mode. Default is (0.1, 6) sigmas.
        theta_range : tuple of floats
            Range of theta values in radians for footprint in polar mode. Default is
            (0.05, pi / 2 - 0.05) radians.
        n_r : int
            Number of r values for footprint in polar mode. Default is 10.
        n_theta : int
            Number of theta values for footprint in polar mode. Default is 10.
        x_norm_range : tuple of floats
            Range of x_norm values for footprint in `uniform action grid` mode.
            Default is (0.1, 6) sigmas.
        y_norm_range : tuple of floats
            Range of y_norm values for footprint in `uniform action grid` mode.
            Default is (0.1, 6) sigmas.
        n_x_norm : int
            Number of x_norm values for footprint in `uniform action grid` mode.
            Default is 10.
        n_y_norm : int
            Number of y_norm values for footprint in `uniform action grid` mode.
            Default is 10.
        linear_rescale_on_knobs: list of xt.LinearRescale
            Detuning from listed knobs is evaluated at a given value of the knob
            with the provided step and rescaled to the actual knob value.
            This is useful to avoid artefact from linear coupling or resonances.
            Example:
                ``line.get_footprint(..., linear_rescale_on_knobs=[
                    xt.LinearRescale(knob_name='beambeam_scale', v0=0, dv-0.1)])``
        freeze_longitudinal : bool
            If True, the longitudinal coordinates are frozen during the particles
            matching and the tracking.
        delta0: float
            Initial value of the delta coordinate.
        zeta0: float
            Initial value of the zeta coordinate in meters.

        Returns
        -------
        fp : Footprint
            Footprint object containing footprint data (fp.qx, fp.qy).

        '''

        kwargs = locals()
        kwargs.pop('self')
        kwargs.pop('linear_rescale_on_knobs')

        freeze_longitudinal = kwargs.pop('freeze_longitudinal')
        delta0 = kwargs.pop('delta0')
        zeta0 = kwargs.pop('zeta0')

        if linear_rescale_on_knobs:
            fp = _footprint_with_linear_rescale(line=self, kwargs=kwargs,
                        linear_rescale_on_knobs=linear_rescale_on_knobs,
                        freeze_longitudinal=freeze_longitudinal,
                        delta0=delta0, zeta0=zeta0)
        else:
            fp = Footprint(**kwargs)
            fp._compute_footprint(self,
                freeze_longitudinal=freeze_longitudinal,
                delta0=delta0, zeta0=zeta0)

        return fp

    def get_amplitude_detuning_coefficients(self, nemitt_x=1e-6, nemitt_y=1e-6,
                num_turns=256, a0_sigmas=0.01, a1_sigmas=0.1, a2_sigmas=0.2):

        '''
        Compute the amplitude detuning coefficients (det_xx = dQx / dJx,
        det_yy = dQy / dJy, det_xy = dQx / dJy, det_yx = dQy / dJx) using
        tracking.

        Parameters
        ----------
        nemitt_x : float
            Normalized emittance in the x-plane. Default is 1e-6.
        nemitt_y : float
            Normalized emittance in the y-plane. Default is 1e-6.
        num_turns : int
            Number of turns for tracking. Default is 256.
        a0_sigmas : float
            Amplitude of the first particle (in sigmas). Default is 0.01.
        a1_sigmas : float
            Amplitude of the second particle (in sigmas). Default is 0.1.
        a2_sigmas : float
            Amplitude of the third particle (in sigmas). Default is 0.2.

        Returns
        -------
        det_xx : float
            Amplitude detuning coefficient dQx / dJx.
        det_yy : float
            Amplitude detuning coefficient dQy / dJy.
        det_xy : float
            Amplitude detuning coefficient dQx / dJy.
        det_yx : float
            Amplitude detuning coefficient dQy / dJx.
        '''

        import nafflib as nl

        gemitt_x = (nemitt_x / self.particle_ref._xobject.beta0[0]
                            / self.particle_ref._xobject.gamma0[0])
        gemitt_y = (nemitt_y / self.particle_ref._xobject.beta0[0]
                            / self.particle_ref._xobject.gamma0[0])

        Jx_1 = a1_sigmas**2 * gemitt_x / 2
        Jx_2 = a2_sigmas**2 * gemitt_x / 2
        Jy_1 = a1_sigmas**2 * gemitt_y / 2
        Jy_2 = a2_sigmas**2 * gemitt_y / 2

        particles = self.build_particles(
                            method='4d',
                            zeta=0, delta=0,
                            x_norm=[a1_sigmas, a2_sigmas, a0_sigmas, a0_sigmas],
                            y_norm=[a0_sigmas, a0_sigmas, a1_sigmas, a2_sigmas],
                            nemitt_x=nemitt_x, nemitt_y=nemitt_y)

        self.track(particles,
                        num_turns=num_turns, time=True,
                        turn_by_turn_monitor=True)
        mon = self.record_last_track

        arr2ctx = particles._context.nparray_from_context_array
        assert np.all(arr2ctx(particles.state) > 0)

        qx = np.zeros(4)
        qy = np.zeros(4)

        # remove average in case there is a closed orbit
        mon.x-=mon.x.mean(axis=1,keepdims=True)
        mon.y-=mon.y.mean(axis=1,keepdims=True)

        for ii in range(len(qx)):
            qx[ii] = np.abs(nl.get_tune(mon.x[ii, :]))
            qy[ii] = np.abs(nl.get_tune(mon.y[ii, :]))

        det_xx = (qx[1] - qx[0]) / (Jx_2 - Jx_1)
        det_yy = (qy[3] - qy[2]) / (Jy_2 - Jy_1)
        det_xy = (qx[3] - qx[2]) / (Jy_2 - Jy_1)
        det_yx = (qy[1] - qy[0]) / (Jx_2 - Jx_1)

        return {'det_xx': det_xx, 'det_yy': det_yy,
                'det_xy': det_xy, 'det_yx': det_yx}

    def compute_one_turn_matrix_finite_differences(
            self, particle_on_co,
            steps_r_matrix=None,
            start=None, end=None,
            num_turns=1,
            element_by_element=False, only_markers=False,
            symmetrize=False,
            include_collective=False):

        '''Compute the one turn matrix using finite differences.

        Parameters
        ----------
        particle_on_co : Particle
            Particle at the closed orbit.
        steps_r_matrix : float
            Step size for finite differences. In not given, default step sizes
            are used.
        start : str
            Optional. It can be used to find the periodic solution for a
            portion of the line.
        end : str
            Optional. It can be used to find the periodic solution for a
            portion of the line.

        Returns
        -------
        one_turn_matrix : np.ndarray
            One turn matrix.

        '''

        self._check_valid_tracker()

        if self.iscollective and not include_collective:
            log.warning(
                'The tracker has collective elements.\n'
                'In the twiss computation collective elements are'
                ' replaced by drifts')
            line = self._get_non_collective_line()
        else:
            line = self

        return compute_one_turn_matrix_finite_differences(line, particle_on_co,
                        steps_r_matrix, start=start, end=end,
                        num_turns=num_turns,
                        element_by_element=element_by_element,
                        only_markers=only_markers,
                        symmetrize=symmetrize)

    def get_non_linear_chromaticity(self,
                        delta0_range=(-1e-3, 1e-3), num_delta=5, fit_order=3, **kwargs):

        '''Get non-linear chromaticity for given range of delta values

        Parameters
        ----------
        delta0_range : tuple of float
            Range of delta values for chromaticity computation.
        num_delta : int
            Number of delta values for chromaticity computation.
        kwargs : dict
            Additional arguments to be passed to the twiss.

        Returns
        -------
        chromaticity : Table
            Table containing the non-linear chromaticity information.

        '''

        return get_non_linear_chromaticity(self, delta0_range, num_delta,
                                           fit_order, **kwargs)

    def get_length(self):

        '''Get total length of the line'''

        ll = 0
        for ee in self.elements:
            if _is_thick(ee, self):
                this_length = _length(ee, self)
                ll += this_length

        return ll

    def get_s_elements(self, mode="upstream"):

        '''Get s position for all elements

        Parameters
        ----------

        mode : str
            "upstream" or "downstream" (default: "upstream")

        Returns
        -------
        s : list of float
            s position for all elements
        '''

        return self.get_s_position(mode=mode)

    def get_s_position(self, at_elements=None, mode="upstream"):

        '''Get s position for given elements

        Parameters
        ----------
        at_elements : str or list of str
            Name of the element(s) to get s position for (default: all elements)
        mode : str
            "upstream" or "downstream" (default: "upstream")

        Returns
        -------
        s : float or list of float
            s position for given element(s)
        '''

        assert mode in ["upstream", "downstream"]
        s_prev = 0.
        s = []
        for ee in self.elements:
            if mode == "upstream":
                s.append(s_prev)
            if _is_thick(ee, line=self):
                this_length = _length(ee, self)
                s_prev += this_length
            if mode == "downstream":
                s.append(s_prev)

        if at_elements is not None:
            if np.isscalar(at_elements):
                if isinstance(at_elements, str):
                    assert at_elements in self.element_names
                    idx = self.element_names.index(at_elements)
                else:
                    idx = at_elements
                return s[idx]
            else:
                assert all([nn in self.element_names for nn in at_elements])
                return [s[self.element_names.index(nn)] for nn in at_elements]
        else:
            return s

    def _elements_intersecting_s(
            self,
            s: List[float],
            s_tol=1e-6,
    ) -> Dict[str, List[float]]:
        """Given a list of s positions, return a list of elements 'cut' by s.

        Arguments
        ---------
        s
            A list of s positions.
        s_tol
            Tolerance used when checking if s falls inside an element, or
            at its edge. Defaults to 1e-6.

        Returns
        -------
        A dictionary, where the keys are the names of the intersected elements,
        and the value for each key is a list of s positions (offset to be
        relative to the start of the element) corresponding to the 'cuts'.
        The structure is ordered such that the cuts are sequential.
        """
        cuts_for_element = defaultdict(list)

        tt = self.get_table()
        all_s_positions = tt.s
        all_s_iter = iter(zip(all_s_positions, tt.name))
        current_s_iter = iter(sorted(set(s)))

        try:
            start, name = next(all_s_iter)
            current_s = next(current_s_iter)

            while True:
                element = self[name]
                if not _is_thick(element, self):
                    start, name = next(all_s_iter)
                    continue

                if np.isclose(current_s, start, atol=s_tol, rtol=0):
                    current_s = next(current_s_iter)
                    continue

                end = start + _length(element, self)
                if np.isclose(current_s, end, atol=s_tol, rtol=0):
                    current_s = next(current_s_iter)
                    continue

                if start < current_s < end:
                    cuts_for_element[name].append(current_s - start)
                    current_s = next(current_s_iter)
                    continue
                if current_s < start:
                    current_s = next(current_s_iter)
                    continue
                if end < current_s:
                    start, name = next(all_s_iter)
                    continue
        except StopIteration:
            # We have either exhausted `s` or the line
            # Do we want to raise an error if `s` was not exhausted?
            pass

        return cuts_for_element

    def cut_at_s(self, s: List[float], s_tol=1e-6, return_slices=False):
        """Slice the line so that positions in s never fall inside an element."""

        if not self._has_valid_tracker():
            self.build_tracker(compile=False) # To resolve replicas and slices

        self.discard_tracker()

        cuts_for_element = self._elements_intersecting_s(s, s_tol=s_tol)
        strategies = [Strategy(None)]  # catch-all, ignore unaffected elements

        for name, cuts in cuts_for_element.items():
            scheme = Custom(at_s=cuts, mode='thick')
            strategy = Strategy(scheme, name=name, exact=True)
            strategies.append(strategy)

        slicer = Slicer(self, slicing_strategies=strategies)
        slices = slicer.slice_in_place()

        if return_slices:
            return slices

    def append(self, what, obj=None):

        """
        Append elements to the line.

        Parameters
        ----------
        what : str, Line or Iterable
            Element(s) to be appended. Can be a list of `Place` objects specifying
            the location of each insertion.
        obj : object (optional)
            Object to be appended (if not already present in the environment).
            It can be specified only when `what` is a string.

        Examples
        --------

        .. code-block:: python

            ## Appending elements from the environment

            # Create a set of new elements to be placed
            env.new('s1', xt.Sextupole, length=0.1, k2=0.2)
            env.new('s2', xt.Sextupole, length=0.1, k2=-0.2)
            env.new('m1', xt.Marker)
            env.new('m2', xt.Marker)
            env.new('m3', xt.Marker)

            # Insert the new elements in the line
            line.append(['m1', 's1', 'm2', 's2', 'm3'])

        .. code-block:: python

            ## Appending elements instantiated by the user using the class
            ## constructor

            myoct = xt.Octupole(length=0.1, k3=0.3)
            line.append('o1', myoct)

        """

        self.discard_tracker()

        if not isinstance(what, (str, xt.Line, Iterable)):
            raise ValueError('The appended object must be defined by a string or Line.')

        if obj is not None:
            assert isinstance(what, str)
            self.env.element_dict[what] = obj

        if isinstance(what, str) in self.element_dict:
            # Is an element and not a line or an iterable
            self.element_names.append(what)
            return

        if not isinstance(what, Iterable) or isinstance(what, str):
            what = [what]

        for item in what:
            if item.__class__.__name__.startswith('Place'):
                raise ValueError('Cannot append a Place object')

        ln_to_append = self.env.new_line(components = what)
        ln_extended = self + ln_to_append

        self.element_names.clear()
        self.element_names.extend(ln_extended.element_names)

    def insert(self, what, obj=None, at=None, from_=None, anchor=None,
               from_anchor=None, s_tol=1e-10):

        """
        Insert elements in the line.

        Parameters
        ----------
        what : str, Line or Iterable
            Element(s) to be inserted. Can be a list of `Place` objects specifying
            the location of each insertion.
        obj : object (optional)
            Object to be inserted (if not already present in the environment).
            It can be specified only when `what` is a string.
        at : str or float (optional)
            Location of the insertion (s position). It can be absolute or relative
            to another element (specified by `from_`).
        from_ : str (optional)
            Element with respect to which `at` is defined.
        anchor : str (optional)
            Location within the inserted element for which `at` is defined.
            It can be 'start', 'end' or 'center'. Default is 'center'.
        from_anchor : str (optional)
            Location within the element specified by `from_` for which `at` is defined.
            It can be 'start', 'end' or 'center'. Default is 'center'.

        Example
        -------

        .. code-block:: python

            ## Insertion of elements from the environment

            # Create a set of new elements to be placed
            env.new('s1', xt.Sextupole, length=0.1, k2=0.2)
            env.new('s2', xt.Sextupole, length=0.1, k2=-0.2)
            env.new('m1', xt.Marker)
            env.new('m2', xt.Marker)
            env.new('m3', xt.Marker)

            # Insert the new elements in the line
            line.insert([
                env.place('s1', at=5.),
                env.place('s2', anchor='end', at=-5., from_='start@q1'),
                env.place(['m1', 'm2'], at='start@m0'),
                env.place('m3', at='end@m0'),
                ])

        .. code-block:: python

            ## Insertion of elements instantiated by the user using the class
            ## constructor

            # Instantiate elements using the class directly
            mysext =  xt.Sextupole(length=0.1, k2=0.2)
            myaperture =  xt.LimitEllipse(a=0.01, b=0.02)

            # Insert the element in the line and, contextually, define its name:
            line.insert('s1', mysext, at=5., from_='q1')

            # Alternatively, add the element to the environment and then do the insertion:
            env.elements['ap1'] = myaperture
            line.insert('ap1', at='start@q0')

        """


        self.discard_tracker()
        env = self.env

        _all_places = xt.environment._all_places
        _resolve_s_positions = xt.environment._resolve_s_positions
        _flatten_components = xt.environment._flatten_components
        _sort_places = xt.environment._sort_places
        _generate_element_names_with_drifts = xt.environment._generate_element_names_with_drifts

        need_place_instantiation = False
        for nn, vv in {'at': at, 'from_': from_, 'anchor': anchor,
                       'from_anchor': from_anchor}.items():
            if vv is not None:
                if (not isinstance(what, (str, xt.Line, Iterable))
                    and not all(isinstance(item, str) for item in what)):
                    raise ValueError(f'The inserted object must be defined by a string '
                                 f'or Line if `{nn}` is provided.')
                need_place_instantiation = True

        if need_place_instantiation:
            what = self.env.place(what, obj=obj, at=at, from_=from_, anchor=anchor,
                                  from_anchor=from_anchor)

        if not isinstance(what, Iterable):
            what = [what]

        # Resolve s positions of insertions and sort them
        what = _flatten_components(what)
        what = _all_places(what)
        what = [ww.copy() for ww in what]

        tt = self.get_table()
        line_places = []
        for nn, enn in zip(tt.name, tt.env_name):
            if nn == '_end_point':
                continue
            line_places.append(env.place(enn, at=tt['s_center', nn]))

        seq_all_places = line_places + what
        mask_insertions = np.array([pp in what for pp in seq_all_places])
        tab_all_unsorted = _resolve_s_positions(seq_all_places, env, refer='centre')
        tab_all_unsorted['is_insertion'] = mask_insertions
        tab_all_sorted = _sort_places(tab_all_unsorted)
        tab_insertions = tab_all_sorted.rows[tab_all_sorted.is_insertion]

        # Make cuts
        s_cuts = list(tab_insertions['s_start']) + list(tab_insertions['s_end'])
        s_cuts = list(set(s_cuts))

        self.cut_at_s(s_cuts, s_tol=1e-06, return_slices=True)

        tt_after_cut = self.get_table()
        tt_after_cut['length'] = np.diff(tt_after_cut.s, append=tt_after_cut.s[-1])

        # Identify old elements falling inside the insertions
        idx_remove = []
        for ii in range(len(tab_insertions)):
            s_ins_start = tab_insertions['s_start', ii]
            s_ins_end = tab_insertions['s_end', ii]
            entry_is_inside = ((tt_after_cut.s_start >= s_ins_start - s_tol)
                            & (tt_after_cut.s_start <= s_ins_end - s_tol))
            exit_is_inside = ((tt_after_cut.s_end >= s_ins_start + s_tol)
                            & (tt_after_cut.s_end <= s_ins_end + s_tol))
            thin_at_entry = ((tt_after_cut.s_start >= s_ins_start - s_tol)
                            & (tt_after_cut.s_end <= s_ins_start + s_tol))
            thin_at_exit = ((tt_after_cut.s_start >= s_ins_end - s_tol)
                        & (tt_after_cut.s_end <= s_ins_end + s_tol))
            remove = (entry_is_inside | exit_is_inside) & (~thin_at_entry) & (~thin_at_exit)
            idx_remove.extend(list(np.where(remove)[0]))

        mask_keep = np.ones(len(tt_after_cut), dtype=bool)
        mask_keep[idx_remove] = False
        tt_keep = tt_after_cut.rows[mask_keep]
        tt_keep['from_'] = np.array([None] * len(tt_keep))
        tt_keep['from_anchor'] = np.array([None] * len(tt_keep))
        assert tt_keep.name[-1] == '_end_point'
        tt_keep = tt_keep.rows[:-1]

        # Unsorted table with all elements for the new line
        tab_unsorted_with_insertions = xt.Table.concatenate([tt_keep, tab_insertions])

        # Sort elements
        tab_sorted = _sort_places(tab_unsorted_with_insertions,
                                  allow_non_existent_from=True # If from_ is removed s only is conisiderer
                                                               # (right order comes form previous sorting,
                                                               # (done before removing elements)
        )
        element_names = _generate_element_names_with_drifts(self.env, tab_sorted, s_tol=s_tol)

        # Update line
        self.element_names.clear()
        self.element_names.extend(element_names)

    def remove(self, name, s_tol=1e-10):

        """
        Remove an element from the line. If the element is thick, it is replaced
        by a drift.

        Parameters
        ----------
        name : str
            Name of the element to be removed.
        s_tol : float (optional)
            If the element is shorter than `s_tol`, it is removed without creating
            a replacement drift. Default is 1e-10.

        """

        self.discard_tracker()

        tt_remove = self._name_match(name)

        mask_thick = tt_remove.isthick & (tt_remove.s_end - tt_remove.s_start > s_tol)
        if mask_thick.any():
            tt_remove_thick = tt_remove.rows[mask_thick]
        else:
            tt_remove_thick = None

        if mask_thick.all():
            tt_remove_thin = None
        else:
            tt_remove_thin = tt_remove.rows[~mask_thick]

        # Replace thick with drifts
        if tt_remove_thick:
            for ii in range(len(tt_remove_thick)):
                ll = tt_remove_thick['s_end', ii] - tt_remove_thick['s_start', ii]
                idx = tt_remove_thick['idx', ii]
                new_name = self.env._get_a_drift_name()
                self.env.new(new_name, 'Drift', length=ll)
                self.element_names[idx] = new_name

        # Remove thin elements
        if tt_remove_thin:
            idx_remove = tt_remove_thin['idx']
            self.element_names = [nn for ii, nn in enumerate(self.element_names)
                                if ii not in idx_remove]

    def replace(self, name, new_name, s_tol=1e-10):

        """
        Replace an element in the line with another element having the same length.

        Parameters
        ----------
        name : str
            Name of the element to be replaced.
        new_name : str
            Name of the element to be installed to replace the removed one.
        s_tol : float (optional)
            Tolerance for the length of the elements. If the difference in length
            is larger than `s_tol`, the replacement is not performed and an
            error is raised.error is raised. Default is 1e-10.

        """

        self.discard_tracker()

        tt_replace = self._name_match(name)

        if _is_thick(self.element_dict[new_name], self):
            l_new = _length(self.element_dict[new_name], self)
        else:
            l_new = 0

        for ii in range(len(tt_replace)):
            l_old = tt_replace['s_end', ii] - tt_replace['s_start', ii]
            if np.abs(l_old - l_new) > s_tol:
                raise ValueError(f'Element {name} cannot be replaced by {new_name} '
                             'because of different lengths.')

        for ii in range(len(tt_replace)):
            idx = tt_replace['idx', ii]
            self.element_names[idx] = new_name

    def _name_match(self, name):
        tt = self.get_table()
        tt['idx'] = np.arange(len(tt))

        idx_match_name = tt.rows.indices[name]

        env_name_match = name
        if isinstance(env_name_match, str):
            env_name_match = [env_name_match]
        env_name_match = set(env_name_match)
        mask_env_name = np.array([nn in env_name_match for nn in tt.env_name])

        idx_match_env_name = tt.rows.indices[mask_env_name]
        idx_match_rep = list(idx_match_name) + list(idx_match_env_name)
        idx_match = []
        for ii in idx_match_rep: # I don't use set to do it in order
            if ii not in idx_match:
                idx_match.append(ii)

        if len(idx_match) == 0:
            raise ValueError(f'Element {name} not found in the line.')

        tt_match = tt.rows[idx_match]

        return tt_match

    # To be deprecated in favor of Line.insert
    def insert_element(self, name, element=None, at=None, index=None, at_s=None,
                       s_tol=1e-6):

        """
        NOTE: This method is deprecated. Use `Line.insert` instead.

        Insert an element in the line.

        Parameters
        ----------
        name: str
            Name of the element.
        element: xline.Element, optional
            Element to be inserted. If not given, the element of the given name
            already present in the line is used.
        at: int or string, optional
            Index or name of the element in the line. If `index` is provided, `at_s` must be None.
        at_s: float, optional
            Position of the element in the line in meters. If `at_s` is provided, `index`
            must be None.
        s_tol: float, optional
            Tolerance for the position of the element in the line in meters.
        """

        if at is not None:
            assert index is None
            index = at

        if isinstance(index, str):
            if '::' in index:
               atelem, count= index.split('::')
               try:
                   index= find_index_repeated(atelem, self.element_names, int(count))
               except ValueError:
                    raise ValueError(f'Element {atelem!r} not found in the line.')
            else:
                try:
                    index = self.element_names.index(index)
                except ValueError:
                    raise ValueError(f'Element {index} not found in the line.')

        if element is None:
            if name not in self.element_dict.keys():
                raise ValueError(
                    f'Element {name} not found in the line. You must either '
                    f'give an `element` or a name of an element already '
                    f'present in the line.'
                )
            element = self.element_dict[name]

        if isinstance(element, xd.madxutils.View):
            element = element._get_viewed_object()

        self.discard_tracker()

        assert ((index is not None and at_s is None) or
                (index is None and at_s is not None)), (
                    "Either `at` or `at_s` must be provided"
                )

        if _is_thick(element, self) and np.abs(_length(element, self)) > 0 and at_s is None:
            raise NotImplementedError('Use `at_s` to insert thick elements')

        # Insert by name or index
        if index is not None:
            self.element_names.insert(index, name)
            self.element_dict[name] = element
            return

        # Insert by s position
        s_vect_upstream = np.array(self.get_s_position(mode='upstream'))

        # Shortcut in case ot thin element and no cut needed
        if not _is_thick(element, self) or np.abs(_length(element, self)) == 0:
            i_closest = np.argmin(np.abs(s_vect_upstream - at_s))
            if np.abs(s_vect_upstream[i_closest] - at_s) < s_tol:
                return self.insert_element(
                    index=i_closest, element=element, name=name)

        s_start_ele = at_s
        if _is_thick(element, self) and np.abs(_length(element, self)) > 0:
            s_end_ele = at_s + _length(element, self)
        else:
            s_end_ele = s_start_ele

        self.cut_at_s([s_start_ele, s_end_ele])

        s_vect_upstream = np.array(self.get_s_position(mode='upstream'))
        if _is_thick(element, self) and _length(element, self) > 0:
            i_first_removal = np.where(np.abs(s_vect_upstream - s_start_ele) < s_tol)[0][-1]
            i_last_removal = np.where(np.abs(s_vect_upstream - s_end_ele) < s_tol)[0][0] - 1
            xo.assert_allclose(s_vect_upstream[i_last_removal + 1]
                              - s_vect_upstream[i_first_removal],
                                _length(element, self), atol=2 * s_tol, rtol=0)
            self.element_names[i_first_removal:i_last_removal + 1] = [name]
        else:
            i_closest = np.argmin(np.abs(s_vect_upstream - at_s))
            assert np.abs(s_vect_upstream[i_closest] - at_s) < s_tol
            self.element_names.insert(i_closest, name)

        if element is None:
            assert name in self.element_dict.keys()
        else:
            self.element_dict[name] = element

        return self

    def append_element(self, element, name):

        """
        NOTE: This method is deprecated. Use `Line.append` instead.

        Append element to the end of the lattice

        Parameters
        ----------
        element : object
            Element to append
        name : str
            Name of the element to append
        """

        if isinstance(element, xd.madxutils.View):
            element = element._get_viewed_object()

        self.discard_tracker()
        if element in self.element_dict and element is not self.element_dict[name]:
            raise ValueError('Element already present in the line')
        self.element_dict[name] = element
        self.element_names.append(name)
        return self

    def filter_elements(self, mask=None, exclude_types_starting_with=None):
        """
        Return a new line with only the elements satisfying a given condition.
        Other elements are replaced with Drifts.

        Parameters
        ----------
        mask: list of bool
            A list of booleans with the same length as the line.
            If True, the element is kept, otherwise it is replaced with a Drift.
        exclude_types_starting_with: str
            If not None, all elements whose type starts with the given string
            are replaced with Drifts.

        Returns
        -------

        new_line: Line
            A new line with only the elements satisfying the condition. Other
            elements are replaced with Drifts.

        """

        if mask is None:
            assert exclude_types_starting_with is not None

        if exclude_types_starting_with is not None:
            assert mask is None
            mask = [not(ee.__class__.__name__.startswith(exclude_types_starting_with))
                    for ee in self.elements]

        new_elements = self.element_dict.copy()
        assert len(mask) == len(self.elements)
        for ff, nn in zip(mask, self.element_names):
            if not ff:
                ee = self.element_dict[nn]
                if hasattr(ee, '_buffer'):
                    _buffer = ee._buffer
                else:
                    _buffer = None
                if _is_thick(ee, self) and not _is_drift(ee, self):
                    new_elements[nn] = Drift(
                        length=_length(ee, self), _buffer=_buffer)
                else:
                    new_elements[nn] = Drift(length=0, _buffer=_buffer)

        new_line = self.__class__(elements=new_elements,
                              element_names=self.element_names)
        if self.particle_ref is not None:
            new_line.particle_ref = self.particle_ref.copy()

        if self._has_valid_tracker():
            new_line.build_tracker(_buffer=self._buffer,
                                   track_kernel=self.tracker.track_kernel)
            #TODO: handle config and other metadata

        return new_line

    def cycle(self, index_first_element=None, name_first_element=None,
              inplace=True):

        """
        Cycle the line to start from a given element.

        Parameters
        ----------
        index_first_element: int
            Index of the element to start from
        name_first_element: str
            Name of the element to start from
        inplace: bool
            If True, the line is modified in place. Otherwise, a new line is returned.

        Returns
        -------
        new_line: Line
            A new line with the elements cycled.

        """

        if not inplace:
            raise ValueError('`inplace=False` is not anymore supported')

        if ((index_first_element is not None and name_first_element is not None)
               or (index_first_element is None and name_first_element is None)):
             raise ValueError(
                "Please provide either `index_first_element` or `name_first_element`.")

        if type(index_first_element) is str:
            name_first_element = index_first_element
            index_first_element = None

        if name_first_element is not None:
            n_occurrences = self.element_names.count(name_first_element)
            if n_occurrences == 0:
                raise ValueError(
                    f"{name_first_element} not found in the line.")
            if n_occurrences > 1:
                raise ValueError(
                    f"{name_first_element} occurs more than once in the line.")
            index_first_element = self.element_names.index(name_first_element)

        new_element_names = (list(self.element_names[index_first_element:])
                             + list(self.element_names[:index_first_element]))

        has_valid_tracker = self._has_valid_tracker()
        if has_valid_tracker:
            buffer = self._buffer
            track_kernel = self.tracker.track_kernel
        else:
            buffer = None
            track_kernel = None

        if inplace:
            self.unfreeze()
            self.element_names = new_element_names
            new_line = self
        else:
            new_line = self.__class__(
                elements=self.element_dict,
                element_names=new_element_names,
                particle_ref=self.particle_ref,
            )

        if has_valid_tracker:
            new_line.build_tracker(_buffer=buffer,
                                   track_kernel=track_kernel)
            #TODO: handle config and other metadata

        return new_line

    def freeze_energy(self, state=True, force=False):

        """
        Freeze energy in tracked Particles objects.

        Parameters
        ----------
        state: bool
            If True, energy is frozen. If False, it is unfrozen.

        """

        assert state in (True, False)
        if not force:
            assert self.iscollective is False, ('Cannot freeze energy '
                            'in collective mode (not yet implemented)')
        if state:
            self.freeze_vars(xt.Particles.part_energy_varnames())
        else:
            self.unfreeze_vars(xt.Particles.part_energy_varnames())

    def _energy_is_frozen(self):
        for vn in xt.Particles.part_energy_varnames():
            flag_name = f'FREEZE_VAR_{vn}'
            if flag_name not in self.config or self.config[flag_name] == False:
                return False
        return True

    def freeze_longitudinal(self, state=True):

        """
        Freeze longitudinal coordinates in tracked Particles objects.

        Parameters
        ----------
        state: bool
            If True, longitudinal coordinates are frozen. If False, they are unfrozen.

        """

        assert state in (True, False)
        assert self.iscollective is False, ('Cannot freeze longitudinal '
                        'variables in collective mode (not yet implemented)')
        if state:
            self.freeze_vars(xt.Particles.part_energy_varnames() + ['zeta'])
        else:
            self.unfreeze_vars(xt.Particles.part_energy_varnames() + ['zeta'])

    def freeze_vars(self, variable_names):

        """
        Freeze variables in tracked Particles objects.

        Parameters
        ----------
        variable_names: list of str
            List of variable names to freeze.

        """

        for name in variable_names:
            self.config[f'FREEZE_VAR_{name}'] = True

    def _var_is_frozen(self, variable_name):
        return self.config[f'FREEZE_VAR_{variable_name}'] == True

    def unfreeze_vars(self, variable_names):

        """
        Unfreeze variables in tracked Particles objects.

        Parameters
        ----------
        variable_names: list of str
            List of variable names to unfreeze.

        """

        for name in variable_names:
            self.config[f'FREEZE_VAR_{name}'] = False

    def configure_bend_model(
            self,
            core=None,
            edge=None,
            num_multipole_kicks=None,
            integrator=None,
    ):

        """
        Configure the method used to track bends.

        See documentation of ``xt.Bend`` for more details on the values of the
        models and schemes used below.

        Parameters
        ----------
        core: str
            Model to be used for the thick bend cores. Can be 'adaptive',
            'full', 'bend-kick-bend', 'rot-kick-rot', 'mat-kick-mat',
            'drift-kick-drift-exact', or 'drift-kick-drift-expanded'.
        edge: str
            Model to be used for the bend edges. Can be 'linear', 'full',
            'dipole-only' or 'suppressed'.
        num_multipole_kicks: int
            Number of multipole kicks to consider.
        integrator: str
            Integration scheme to be used. Can be 'adaptive', 'teapot',
            'yoshida4', or 'uniform'.
        """

        if core is not None and core not in _MODEL_TO_INDEX_CURVED:
            raise ValueError(f'Unknown bend model {core}')

        if edge is not None and edge not in _EDGE_MODEL_TO_INDEX:
            raise ValueError(f'Unknown bend edge model {edge}')

        for ee in self.element_dict.values():
            if core is not None and isinstance(ee, (xt.Bend, xt.RBend)):
                ee.model = core

            if edge is not None and isinstance(ee, xt.DipoleEdge):
                ee.model = edge if not edge == 'dipole-only' else 'full'

            if edge is not None and isinstance(ee, (xt.Bend, xt.RBend)):
                ee.edge_entry_model = edge
                ee.edge_exit_model = edge

            if num_multipole_kicks is not None:
                ee.num_multipole_kicks = num_multipole_kicks

            if integrator is not None:
                ee.integrator = integrator

    def _configure_mult(
            self,
            element_type,
            model=None,
            edge: Optional[Literal['full']] = None,
            num_multipole_kicks: Optional[int] = None,
            integrator: Optional[str] = None,
    ):
        """Configure fringes on elements of a given type.

        Parameters
        ----------
        edge: str
            None to disable, 'full' to enable.
        num_multipole_kicks: int
            Number of multipole kicks to consider.
        integrator: str
            Integration scheme to be used. Can be 'adaptive', 'teapot',
            'yoshida4', or 'uniform'.
        """
        if edge not in [None, 'full', 'suppressed']:
            raise ValueError(f'Unknown edge model {edge}: only None or '
                             f'"full" are supported.')

        enable_fringes = edge == 'full'

        for ee in self.element_dict.values():
            if not isinstance(ee, element_type):
                continue
            if edge is not None:
                ee.edge_entry_active = enable_fringes
                ee.edge_exit_active = enable_fringes
            if num_multipole_kicks is not None:
                ee.num_multipole_kicks = num_multipole_kicks
            if integrator is not None:
                ee.integrator = integrator
            if model is not None:
                ee.model = model

    def configure_quadrupole_model(self,
            model: Optional[str] = None,
            edge: Optional[Literal['full']] = None,
            num_multipole_kicks: Optional[int] = None,
            integrator: Optional[str] = None,
    ):
        self._configure_mult(
            xt.Quadrupole,
            model=model,
            edge=edge,
            num_multipole_kicks=num_multipole_kicks,
            integrator=integrator,
        )

    def configure_sextupole_model(
            self,
            model: Optional[str] = None,
            edge: Optional[Literal['full']] = None,
            num_multipole_kicks: Optional[int] = None,
            integrator: Optional[str] = None,
    ):
        self._configure_mult(
            xt.Sextupole,
            model=model,
            edge=edge,
            num_multipole_kicks=num_multipole_kicks,
            integrator=integrator,
        )

    def configure_octupole_model(
            self,
            model: Optional[str] = None,
            edge: Optional[Literal['full']] = None,
            num_multipole_kicks: Optional[int] = None,
            integrator: Optional[str] = None,
    ):
        self._configure_mult(
            xt.Octupole,
            model=model,
            edge=edge,
            num_multipole_kicks=num_multipole_kicks,
            integrator=integrator,
        )

    def configure_radiation(self, model=None, model_beamstrahlung=None,
                            model_bhabha=None, mode='deprecated'):

        """
        Configure radiation within the line.

        Parameters
        ----------
        model: str
            Radiation model to use. Can be 'mean', 'quantum' or None.
        model_beamstrahlung: str
            Beamstrahlung model to use. Can be 'mean', 'quantum' or None.
        model_bhabha: str
            Bhabha model to use. Can be 'quantum' or None.
        """

        if mode != 'deprecated':
            raise NameError('mode is deprecated, use model instead')

        if not self._has_valid_tracker():
            self.build_tracker(compile=False)

        assert model in [None, 'mean', 'quantum']
        assert model_beamstrahlung in [None, 'mean', 'quantum']
        assert model_bhabha in [None, 'quantum']

        if model == 'mean':
            radiation_flag = 1
            self._radiation_model = 'mean'
        elif model == 'quantum':
            radiation_flag = 2
            self._radiation_model = 'quantum'
        else:
            radiation_flag = 0
            self._radiation_model = None

        if model_beamstrahlung == 'mean':
            beamstrahlung_flag = 1
            self._beamstrahlung_model = 'mean'
        elif model_beamstrahlung == 'quantum':
            beamstrahlung_flag = 2
            self._beamstrahlung_model = 'quantum'
        else:
            beamstrahlung_flag = 0
            self._beamstrahlung_model = None

        if model_bhabha == 'quantum':
            bhabha_flag = 1
            self._bhabha_model = 'quantum'
        else:
            bhabha_flag = 0
            self._bhabha_model = None

        for kk, ee in self.element_dict.items():
            if hasattr(ee, 'radiation_flag'):
                ee.radiation_flag = radiation_flag

        for kk, ee in self.element_dict.items():
            if hasattr(ee, 'flag_beamstrahlung'):
                ee.flag_beamstrahlung = beamstrahlung_flag
            if hasattr(ee, 'flag_bhabha'):
                ee.flag_bhabha = bhabha_flag

        if radiation_flag == 2 or beamstrahlung_flag == 2 or bhabha_flag == 1:
            self._needs_rng = True

        self.config.XFIELDS_BB3D_NO_BEAMSTR = (beamstrahlung_flag == 0)
        self.config.XFIELDS_BB3D_NO_BHABHA = (bhabha_flag == 0)

        self._update_synrad_compile_flag()

    def _update_synrad_compile_flag(self):

        if self._radiation_model or self._spin_model:
            self.config.XTRACK_MULTIPOLE_NO_SYNRAD = False
        else:
            self.config.XTRACK_MULTIPOLE_NO_SYNRAD = True

    def configure_spin(self, spin_model=None):

        """
        Configure the spin model for the line.

        Parameters
        ----------
        spin_model: str
            Spin model to use. Can be None, 'auto', 'True', 'False'
        """

        assert spin_model in [None, 'auto', 'True', 'False']
        if spin_model is False:
            spin_model = None
        if spin_model is True:
            spin_model = 'auto'

        self._spin_model = spin_model

        self._update_synrad_compile_flag()

    def configure_intrabeam_scattering(
        self, element = None,
        update_every: int = None,
        **kwargs,
    ) -> None:
        """
        Configures the IBS kick element in the line for tracking.

        Notes
        -----
            This **should be** one of the last steps taken before tracking.
            At the very least, if steps are taken that change the lattice's
            optics after this configuration, then this function should be
            called once again.

        Parameters
        ----------
        line : xtrack.Line
            The line in which the IBS kick element was inserted.
        element : IBSKick, optional
            If provided, the element is first inserted in the line,
            before proceeding to configuration. In this case the keyword
            arguments are passed on to the `line.insert_element` method.
        update_every : int
            The frequency at which to recompute the kick coefficients, in
            number of turns. They will be computed at the first turn of
            tracking, and then every `update_every` turns afterwards.
        **kwargs : dict, optional
            Required if an element is provided. Keyword arguments are
            passed to the `line.insert_element()` method according to
            `line.insert_element(element=element, **kwargs)`.

        Raises
        ------
        ImportError
            If the xfields package is not installed, with a sufficiently
            recent version.
        AssertionError
            If the provided `update_every` is not a positive integer.
        AssertionError
            If more than one IBS kick element is found in the line.
        AssertionError
            If the element is an `IBSSimpleKick` and the line is operating
            below transition energy.
        """
        try:
            from xfields.ibs import configure_intrabeam_scattering
        except ImportError as error:
            raise ImportError("Please install xfields to use this feature.") from error
        configure_intrabeam_scattering(
            self, element=element, update_every=update_every, **kwargs
        )

    def compensate_radiation_energy_loss(self, delta0='zero_mean', rtol_eneloss=1e-10,
                                    max_iter=100, **kwargs):

        """
        Compensate beam energy loss from synchrotron radiation by configuring
        RF cavities and Multipole elements (tapering).

        Parameters
        ----------
        delta0: float
            Initial energy deviation. If `delta0='zero_mean'` is specified, the
            compensation is done such that the mean energy deviation along the
            ring is zero.
        rtol_eneloss: float
            Relative tolerance on energy loss.
        max_iter: int
            Maximum number of iterations.
        kwargs: dict
            Additional keyword arguments passed to the twiss method.

        """

        all_kwargs = locals().copy()
        all_kwargs.pop('self')
        all_kwargs.pop('kwargs')
        all_kwargs.update(kwargs)
        self._check_valid_tracker()
        compensate_radiation_energy_loss(self, **all_kwargs)

    def optimize_for_tracking(self, compile=True, verbose=True, keep_markers=False):

        """
        Optimize the line for tracking by removing inactive elements and
        merging consecutive elements where possible. Deferred expressions are
        disabled.

        Parameters
        ----------
        compile: bool
            If True (default), the tracker is recompiled.
        verbose: bool
            If True (default), print information about the optimization.
        keep_markers: bool or list of str
            If True, all markers are kept.

        """

        if self.iscollective:
            raise NotImplementedError("Optimization is not implemented for "
                                      "collective trackers")

        self.tracker.track_kernel.clear() # Remove all kernels

        if verbose: _print("Disable xdeps expressions")
        self._var_management = None # Disable expressions
        if hasattr(self, '_in_multiline') and self._in_multiline is not None:
            self._in_multiline._var_sharing = None

        buffer = self._buffer
        io_buffer = self.tracker.io_buffer

        # Unfreeze the line
        self.discard_tracker()

        if verbose: _print("Replace slices with equivalent elements")
        self._replace_with_equivalent_elements()

        if keep_markers is True:
            if verbose: _print('Markers are kept')
        elif keep_markers is False:
            if verbose: _print("Remove markers")
            self.remove_markers()
        else:
            if verbose: _print('Keeping only selected markers')
            self.remove_markers(keep=keep_markers)

        if verbose: _print("Remove inactive multipoles")
        self.remove_inactive_multipoles()

        if verbose: _print("Merge consecutive multipoles")
        self.merge_consecutive_multipoles()

        if verbose: _print("Remove redundant apertures")
        self.remove_redundant_apertures()

        if verbose: _print("Remove zero length drifts")
        self.remove_zero_length_drifts()

        if verbose: _print("Merge consecutive drifts")
        self.merge_consecutive_drifts()

        if verbose: _print("Use simple bends")
        self.use_simple_bends()

        if verbose: _print("Use simple quadrupoles")
        self.use_simple_quadrupoles()

        if verbose: _print("Rebuild tracker data")
        self.build_tracker(_buffer=buffer, io_buffer=io_buffer)

        self.use_prebuilt_kernels = False

        if compile:
            _ = self.tracker.get_track_kernel_and_data_for_present_config()

    def start_internal_logging_for_elements_of_type(self,
                                                    element_type, capacity):
        """
        Start internal logging for all elements of a given type.

        Parameters
        ----------
        element_type: str
            Type of the elements for which internal logging is started.
        capacity: int
            Capacity of the internal record.

        Returns
        -------
        record: Record
            Record object containing the elements internal logging.

        """
        self._check_valid_tracker()
        return start_internal_logging_for_elements_of_type(self.tracker,
                                                    element_type, capacity)

    def stop_internal_logging_for_all_elements(self, reinitialize_io_buffer=False):
        """
        Stop internal logging for all elements.

        Parameters
        ----------
        reinitialize_io_buffer: bool
            If True, the IO buffer is reinitialized (default: False).

        """
        self._check_valid_tracker()
        stop_internal_logging(elements=self.elements)

        if reinitialize_io_buffer:
            self.tracker._init_io_buffer()

    def stop_internal_logging_for_elements_of_type(self, element_type):

        """
        Stop internal logging for all elements of a given type.

        Parameters
        ----------
        element_type: str
            Type of the elements for which internal logging is stopped.

        """

        self._check_valid_tracker()
        stop_internal_logging_for_elements_of_type(self.tracker, element_type)

    def extend_knl_ksl(self, order, element_names=None):

        """
        Extend the order of the knl and ksl attributes of the elements.

        Parameters
        ----------
        order: int
            New order of the knl and ksl attributes.
        element_names: list of str
            Names of the elements to extend. If None, all elements having `knl`
            and `ksl` attributes are extended.

        """

        self.discard_tracker()

        if element_names is None:
            element_names = []
            for nn in self.element_names:
                if hasattr(self.get(nn), 'knl'):
                    element_names.append(nn)

        for nn in element_names:
            if self.get(nn).order > order:
                raise ValueError(f'Order of element {nn} is smaller than {order}')

        for nn in element_names:
            ee = self.get(nn)

            if ee.order == order:
                continue

            new_knl = [vv for vv in ee.knl] + [0] * (order - len(ee.knl) + 1)
            new_ksl = [vv for vv in ee.ksl] + [0] * (order - len(ee.ksl) + 1)

            dct = ee.to_dict()
            dct.pop('order', None)
            dct['knl'] = new_knl
            dct['ksl'] = new_ksl

            new_ee = ee.__class__.from_dict(dct, _buffer=ee._buffer)
            self.env.elements[nn] = new_ee

    def remove_markers(self, inplace=True, keep=None):
        """
        Remove markers from the line

        Parameters
        ----------
        inplace : bool
            If True, remove markers from the line (default: True)
        keep : str or list of str
            Name of the markers to keep (default: None)
        """
        self._frozen_check()

        if keep is None:
            keep = []
        elif isinstance(keep, str):
            keep = [keep]

        newline = Line(elements=[], element_names=[])

        for ee, nn in zip(self.elements, self.element_names):
            if isinstance(ee, Marker) and nn not in keep:
                continue
            newline.append_element(ee, nn)

        if inplace:
            self.element_names = newline.element_names
            self.element_dict.update(newline.element_dict)
            return self
        else:
            return newline

    def remove_inactive_multipoles(self, inplace=True, keep=None):

        '''
        Remove inactive multipoles from the line

        Parameters
        ----------
        inplace : bool
            If True, remove inactive multipoles from the line (default: True),
            otherwise return a new line.
        keep : str or list of str
            Name of the multipoles to keep (default: None)

        Returns
        -------
        line : Line
            Line with inactive multipoles removed

        '''

        if not _vars_unused(self):
            raise NotImplementedError('`remove_inactive_multipoles` not'
                                      ' available when deferred expressions are'
                                      ' used')

        self._frozen_check()

        if keep is None:
            keep = []
        elif isinstance(keep, str):
            keep = [keep]

        newline = Line(elements=[], element_names=[])

        for ee, nn in zip(self.elements, self.element_names):
            if isinstance(ee, Multipole) and nn not in keep:
                ctx2np = ee._context.nparray_from_context_array
                aux = ([ee.hxl]
                        + list(ctx2np(ee.knl)) + list(ctx2np(ee.ksl)))
                if np.sum(np.abs(np.array(aux))) == 0.0:
                    continue
            newline.append_element(ee, nn)

        if inplace:
            self.element_names = newline.element_names
            self.element_dict.update(newline.element_dict)
            return self
        else:
            return newline

    def remove_zero_length_drifts(self, inplace=True, keep=None):

        '''
        Remove zero length drifts from the line

        Parameters
        ----------
        inplace : bool
            If True, remove zero length drifts from the line (default: True),
            otherwise return a new line.
        keep : str or list of str
            Name of the drifts to keep (default: None)

        Returns
        -------
        line : Line
            Line with zero length drifts removed

        '''

        if not _vars_unused(self):
            raise NotImplementedError('`remove_zero_length_drifts` not'
                                      ' available when deferred expressions are'
                                      ' used')

        self._frozen_check()

        if keep is None:
            keep = []
        elif isinstance(keep, str):
            keep = [keep]

        newline = Line(elements=[], element_names=[])

        for ee, nn in zip(self.elements, self.element_names):
            if _is_drift(ee, self) and nn not in keep:
                if _length(ee, self) == 0.0:
                    continue
            newline.append_element(ee, nn)

        if inplace:
            self.element_names = newline.element_names
            self.element_dict.update(newline.element_dict)
            return self
        else:
            return newline

    def merge_consecutive_drifts(self, inplace=True, keep=None):

        '''
        Merge consecutive drifts into a single drift

        Parameters
        ----------
        inplace : bool
            If True, merge consecutive drifts in the line (default: True),
            otherwise return a new line.
        keep : str or list of str
            Name of the drifts to keep (default: None)

        Returns
        -------
        line : Line
            Line with consecutive drifts merged

        '''

        assert inplace is True, 'Only inplace is supported for now'

        self._frozen_check()
        self.replace_all_repeated_elements()

        if keep is None:
            keep = []
        elif isinstance(keep, str):
            keep = [keep]

        newline = Line(elements=[], element_names=[])

        for ii, (ee, nn) in enumerate(zip(self.elements, self.element_names)):
            if ii == 0:
                newline.append_element(ee, nn)
                continue

            this_ee = ee if inplace else ee.copy()
            if _is_drift(ee, self) and not nn in keep:
                prev_nn = newline.element_names[-1]
                prev_ee = newline.element_dict[prev_nn]
                if _is_drift(prev_ee, self) and not prev_nn in keep:
                    prev_ee.length += ee.length
                else:
                    newline.append_element(this_ee, nn)
            else:
                newline.append_element(this_ee, nn)

        if inplace:
            self.element_names = newline.element_names
            self.element_dict.update(newline.element_dict)
            return self
        else:
            return newline

    def remove_redundant_apertures(self, inplace=True, keep=None,
                                  drifts_that_need_aperture=[]):

        '''
        Remove redundant apertures from the line

        Parameters
        ----------
        inplace : bool
            If True, remove redundant apertures from the line (default: True),
            otherwise return a new line.
        keep : str or list of str
            Name of the apertures to keep (default: None)
        drifts_that_need_aperture : list of str
            Names of drifts that need an aperture (default: [])

        Returns
        -------
        line : Line
            Line with redundant apertures removed

        '''

        if not inplace:
            raise NotImplementedError('`remove_redundant_apertures` only'
                                      ' available for inplace operation')

        # For every occurence of three or more apertures that are the same,
        # only separated by Drifts or Markers, this script removes the
        # middle apertures
        # TODO: this probably actually works, but better be safe than sorry
        if not _vars_unused(self):
            raise NotImplementedError('`remove_redundant_apertures` not'
                                      ' available when deferred expressions are'
                                      ' used')

        self._frozen_check()

        if keep is None:
            keep = []
        elif isinstance(keep, str):
            keep = [keep]

        aper_to_remove = []
        # current aperture in loop
        aper_0  = None
        # previous aperture in loop (-1)
        aper_m1 = None
        # aperture before previous in loop (-2)
        aper_m2 = None

        for ee, nn in zip(self.elements, self.element_names):
            if _is_aperture(ee, self):
            # We encountered a new aperture, shift all previous
                aper_m2 = aper_m1
                aper_m1 = aper_0
                aper_0  = nn
            elif ((not isinstance(ee, (Marker)) and not _is_drift(ee, self))
                  or nn in drifts_that_need_aperture):
            # We are in an active element: all previous apertures
            # should be kept in the line
                aper_0  = None
                aper_m1 = None
                aper_m2 = None
            if (aper_m2 is not None
                and _apertures_equal(
                    self.element_dict[aper_0], self.element_dict[aper_m1], self)
                and _apertures_equal(
                    self.element_dict[aper_m1], self.element_dict[aper_m2], self)
                ):
                # We found three consecutive apertures (with only Drifts and Markers
                # in between) that are the same, hence the middle one can be removed
                if aper_m1 not in keep:
                    aper_to_remove = [*aper_to_remove, aper_m1]
                    # Middle aperture removed, so the -2 shifts to the -1 position
                    aper_m1 = aper_m2
                    aper_m2 = None

        if inplace:
            newline = self
        else:
            newline = self.copy()

        for name in aper_to_remove:
            newline.element_names.remove(name)

        return newline

    def use_simple_quadrupoles(self):
        '''
        Replace multipoles having only the normal quadrupolar component
        with quadrupole elements. The element is not replaced when synchrotron
        radiation is active.
        '''
        self._frozen_check()

        for name, element in self.element_dict.items():
            if _is_simple_quadrupole(element):
                fast_quad = beam_elements.SimpleThinQuadrupole(
                    knl=element.knl[:2],
                    _context=element._context,
                )
                self.element_dict[name] = fast_quad

    def use_simple_bends(self):
        '''
        Replace multipoles having only the horizontal dipolar component
        with dipole elements. The element is not replaced when synchrotron
        radiation is active.
        '''
        self._frozen_check()

        for name, element in self.element_dict.items():
            if _is_simple_dipole(element):
                fast_di = beam_elements.SimpleThinBend(
                    knl=element.knl[:1],
                    hxl=element.hxl,
                    length=element.length,
                    _context=element._context,
                )
                self.element_dict[name] = fast_di

    def get_elements_of_type(self, types):

        '''Get all elements of given type(s)

        Parameters
        ----------
        types : type or list of types
            Type(s) of elements to get

        Returns
        -------
        elements : list of elements
            List of elements of given type(s)
        names : list of str
            List of names of elements of given type(s)

        '''

        if not hasattr(types, "__iter__"):
            type_list = [types]
        else:
            type_list = types

        names = []
        elements = []
        for ee, nn in zip(self.elements, self.element_names):
            for tt in type_list:
                if isinstance(ee, tt):
                    names.append(nn)
                    elements.append(ee)

        return elements, names

    def check_aperture(self, needs_aperture=[]):

        '''Check that all active elements have an associated aperture.

        Parameters
        ----------
        needs_aperture : list of str
            Names of inactive elements that also need an aperture.

        Returns
        -------
        elements_df : pandas.DataFrame
            DataFrame with information about the apertures associated with
            each active element.
        '''

        elements_df = self.to_pandas()

        elements_df['is_aperture'] = elements_df.name.map(
                lambda nn: nn == '_end_point'
                    or  _is_aperture(self.element_dict[nn], self))

        if not elements_df.name.values[-1] == '_end_point':
            elements_df['is_aperture'][-1] = False

        elements_df['i_aperture_upstream'] = np.nan
        elements_df['s_aperture_upstream'] = np.nan
        elements_df['i_aperture_downstream'] = np.nan
        elements_df['s_aperture_downstream'] = np.nan
        num_elements = len(self.element_names)

        # Elements that don't need aperture
        dont_need_aperture = {name: False for name in elements_df['name']}
        for name in elements_df['name']:
            if name == '_end_point':
                continue
            ee = self.element_dict[name]
            if isinstance(ee, xt.Replica):
                ee = ee.resolve(self)
            if _allow_loss_refinement(ee, self) and not name in needs_aperture:
                dont_need_aperture[name] = True
            if name.endswith('_entry') or name.endswith('_exit'):
                dont_need_aperture[name] = True

            # Correct isthick for elements that need aperture but have zero length.
            # Use-case example: Before collimators are installed as EverestCollimator
            # (or any BaseCollimator element), they are just Markers or Drifts. We
            # want to enforce that they have an aperture when loading the line (when
            # they are still Drifts), so their names are added to 'needs_aperture'.
            # However, it is enough for them to have an upstream aperture as they are
            # at this stage just Markers (and xcoll takes care of providing the down-
            # stream aperture), so we mark them as thin.
            if name in needs_aperture and hasattr(ee, 'length') and _length(ee, self) == 0:
                elements_df.loc[elements_df['name']==name, 'isthick'] = False

        i_prev_aperture = elements_df[elements_df['is_aperture']].index[0]
        i_next_aperture = 0

        for iee in progress(range(i_prev_aperture, num_elements), desc='Checking aperture'):
            if elements_df.loc[iee, 'is_aperture']:
                i_prev_aperture = iee
                continue

            if dont_need_aperture[elements_df.loc[iee, 'name']]:
                continue

            if i_next_aperture < iee:
                for ii in range(iee, num_elements):
                    if elements_df.loc[ii, 'is_aperture']:
                        i_next_aperture = ii
                        break

            elements_df.at[iee, 'i_aperture_upstream'] = i_prev_aperture
            elements_df.at[iee, 'i_aperture_downstream'] = i_next_aperture

            elements_df.at[iee, 's_aperture_upstream'] = elements_df.loc[i_prev_aperture, 's']
            elements_df.at[iee, 's_aperture_downstream'] = elements_df.loc[i_next_aperture, 's']

        # Check for elements missing aperture upstream
        elements_df['misses_aperture_upstream'] = ((elements_df['s_aperture_upstream'] != elements_df['s'])
            & ~(np.isnan(elements_df['i_aperture_upstream'])))

        # Check for elements missing aperture downstream
        s_downstream = elements_df.s.copy()
        df_thick_to_check = elements_df[elements_df['isthick'] & ~(elements_df.i_aperture_upstream.isna())].copy()
        s_downstream.loc[df_thick_to_check.index] += np.array([_length(ee, self) for ee in df_thick_to_check.element])
        elements_df['misses_aperture_downstream'] = (
            (np.abs(elements_df['s_aperture_downstream'] - s_downstream) > 1e-6)
            & ~(np.isnan(elements_df['i_aperture_upstream'])))

        # Flag problems
        elements_df['has_aperture_problem'] = (
            elements_df['misses_aperture_upstream'] | (
                elements_df['isthick'] & elements_df['misses_aperture_downstream']))

        _print('Done checking aperture.           ')

        # Identify issues with apertures associate with thin elements
        df_thin_missing_aper = elements_df[elements_df['misses_aperture_upstream'] & ~elements_df['isthick']]
        _print(f'{len(df_thin_missing_aper)} thin elements miss associated aperture (upstream):')
        _print(pformat(list(df_thin_missing_aper.name)))

        # Identify issues with apertures associate with thick elements
        df_thick_missing_aper = elements_df[
            (elements_df['misses_aperture_upstream'] | elements_df['misses_aperture_downstream'])
            & elements_df['isthick']]
        _print(f'{len(df_thick_missing_aper)} thick elements miss associated aperture (upstream or downstream):')
        _print(pformat(list(df_thick_missing_aper.name)))

        return elements_df

    def merge_consecutive_multipoles(self, inplace=True, keep=None):
        '''
        Merge consecutive multipoles into one multipole.

        Parameters
        ----------
        inplace : bool, optional
            If True, the line is modified in place. If False, a new line is
            returned.
        keep : str or list of str, optional
            Names of elements that should not be merged. If None, no elements
            are kept.

        Returns
        -------
        line : Line
            The modified line.
        '''

        if not _vars_unused(self):
            raise NotImplementedError('`merge_consecutive_multipoles` not'
                                      ' available when deferred expressions are'
                                      ' used')

        self._frozen_check()
        self.replace_all_repeated_elements()

        if keep is None:
            keep = []
        elif isinstance(keep, str):
            keep = [keep]

        newline = Line(elements=[], element_names=[])

        for ee, nn in zip(self.elements, self.element_names):
            if len(newline.element_names) == 0:
                newline.append_element(ee, nn)
                continue

            if isinstance(ee, Multipole) and nn not in keep:
                prev_nn = newline.element_names[-1]
                prev_ee = newline.element_dict[prev_nn]
                if (isinstance(prev_ee, Multipole)
                    and prev_ee.hxl==ee.hxl==0
                    and prev_nn not in keep
                    ):

                    oo=max(len(prev_ee.knl), len(prev_ee.ksl),
                           len(ee.knl), len(ee.ksl))
                    knl=np.zeros(oo,dtype=float)
                    ksl=np.zeros(oo,dtype=float)
                    for ii,kk in enumerate(prev_ee._xobject.knl):
                        knl[ii]+=kk
                    for ii,kk in enumerate(ee._xobject.knl):
                        knl[ii]+=kk
                    for ii,kk in enumerate(prev_ee._xobject.ksl):
                        ksl[ii]+=kk
                    for ii,kk in enumerate(ee._xobject.ksl):
                        ksl[ii]+=kk
                    newee = Multipole(
                            knl=knl, ksl=ksl, hxl=prev_ee.hxl,
                            length=prev_ee.length,
                            radiation_flag=prev_ee.radiation_flag)
                    prev_nn += ('_' + nn)
                    newline.element_dict[prev_nn] = newee
                    newline.element_names[-1] = prev_nn
                else:
                    newline.append_element(ee, nn)
            else:
                newline.append_element(ee, nn)

        if inplace:
            self.element_names = newline.element_names
            self.element_dict.update(newline.element_dict)
            return self
        else:
            return newline

    def get_line_with_second_order_maps(self, split_at):

        '''
        Return a new lines with segments definded by the elements in `split_at`
        replaced by second order maps.

        Parameters
        ----------
        split_at : list of str
            Names of elements at which to split the line.

        Returns
        -------
        line_maps : Line
            Line with segments replaced by second order maps.
        '''

        ele_cut_ext = split_at.copy()
        if self.element_names[0] not in ele_cut_ext:
            ele_cut_ext.insert(0, self.element_names[0])
        if self.element_names[-1] not in ele_cut_ext:
            ele_cut_ext.append(self.element_names[-1])

        ele_cut_sorted = []
        for ee in self.element_names:
            if ee in ele_cut_ext:
                ele_cut_sorted.append(ee)

        elements_map_line = []
        names_map_line = []
        tw = self.twiss()

        for ii in range(len(ele_cut_sorted)-1):
            names_map_line.append(ele_cut_sorted[ii])
            elements_map_line.append(self.get(ele_cut_sorted[ii]))

            smap = xt.SecondOrderTaylorMap.from_line(
                                    self, start=ele_cut_sorted[ii],
                                    end=ele_cut_sorted[ii+1],
                                    twiss_table=tw,
                                    _buffer=self._buffer)
            names_map_line.append(f'map_{ii}')
            elements_map_line.append(smap)

        names_map_line.append(ele_cut_sorted[-1])
        elements_map_line.append(self[ele_cut_sorted[-1]])

        line_maps = Line(elements=elements_map_line, element_names=names_map_line)
        line_maps.particle_ref = self.particle_ref.copy()

        return line_maps

    def target(self, tar, value, **kwargs):

        action = ActionLine(line=self)
        return xt.Target(action=action, tar=tar, value=value, **kwargs)

    def _freeze(self):
        self.element_names = tuple(self.element_names)

    def unfreeze(self):

        # Unfreeze the line. This is useful if you want to modify the line
        # after it has been frozen (most likely by calling `build_tracker`).

        self.discard_tracker()

    def _frozen_check(self):
        if isinstance(self.element_names, tuple):
            raise ValueError(
                'This action is not allowed as the line is frozen! '
                'You can unfreeze the line by calling the `discard_tracker()` method.')

    def transform_compound(self, *args, **kwargs):
        raise NotImplementedError('`transform_compound` is not anymore supported. '
            'Tilt and shifts can be applied directly to the elements.\n'
            'For more details, see: '
            'https://xsuite.readthedocs.io/en/latest/line.html#apply-transformations-tilt-shift-to-elements'
        )

    def mirror(self, inplace=True):
        assert inplace in [True, False]
        if inplace == False:
            out = self.select()
            out.mirror(inplace=True)
            return out
        else:
            self._frozen_check()
            self.element_names = list(reversed(self.element_names))

    def __neg__(self):
        return self.mirror(inplace=False)

    def __rmul__(self, other):
        self._env_if_needed()
        assert isinstance(other, int), 'Only integer multiplication is supported'
        assert other > 0, 'Only positive integer multiplication is supported'
        ele_names = list(self.element_names)
        out = self.env.new_line()
        out.element_names = ele_names * other
        return out

    def __add__(self, other):
        self._env_if_needed
        #assert isinstance(other, Line), 'Only Line can be added to Line'
        assert other.__class__.__name__=="Line", 'Only Line can be added to Line'
        assert other.env is self.env, 'Lines must be in the same environment'
        out = self.env.new_line(
            components=list(self.element_names) + list(other.element_names))
        return out

    def __sub__(self, other):
        return self + (-other)

    def replicate(self, name, mirror=False):

        self._env_if_needed()

        new_element_names = []
        for nn in self.element_names:
            new_nn = nn + '.' + name
            self.element_dict[new_nn] = xt.Replica(nn)
            new_element_names.append(new_nn)

        out = self.env.new_line(components=new_element_names, name=name)

        if mirror:
            out.mirror()

        return out

    def clone(self, name, mirror=False):
        out = self.replicate(name=name, mirror=mirror)
        out.replace_all_replicas()
        return out

    def replace_replica(self, name):
        name_parent = self.element_dict[name].resolve(self, get_name=True)
        self.copy_element_from(name_parent, self, new_name=name)

    def copy_element_from(self, name, source, new_name=None):
        """Copy an element from another environment.

        Parameters
        ----------
        name: str
            Name of the element to copy.
        source: Environment | Line
            Environment or line where the element is located.
        new_name: str, optional
            Rename the element in the new line/environment. If not provided, the
            element is copied with the same name.
        """
        new_name_input = new_name if new_name != name else None
        new_name = new_name or name
        cls = type(source.element_dict[name])

        if (cls not in _ALLOWED_ELEMENT_TYPES_IN_NEW + [xt.DipoleEdge] # No issue in copying DipoleEdge while creating it requires handling properties which are strings.
            and 'ThickSlice' not in cls.__name__ and 'ThinSlice' not in cls.__name__
            and 'DriftSlice' not in cls.__name__):
            raise ValueError(
                f'Only {_STR_ALLOWED_ELEMENT_TYPES_IN_NEW} elements are '
                f'allowed in `copy_from_env` for now.'
            )

        self.element_dict[new_name] = source.element_dict[name].copy()

        pars_with_expr = list(
            source._xdeps_manager.tartasks[source.element_refs[name]].keys())

        formatter = xd.refs.CompactFormatter(scope=None)

        for rr in pars_with_expr:
            # Assign expressions by string to avoid having to deal with the
            # fact that they come from a different manager!
            expr_string = rr._expr._formatted(formatter)

            if isinstance(rr, xd.refs.AttrRef):
                setattr(self[new_name], rr._key, expr_string)
            elif isinstance(rr, xd.refs.ItemRef):
                getattr(self[new_name], rr._owner._key)[rr._key] = expr_string
            else:
                raise ValueError('Only AttrRef and ItemRef are supported for now')

        return new_name

    def replace_all_replicas(self):
        for nn in self.element_names:
            if isinstance(self.element_dict[nn], xt.Replica):
                self.replace_replica(nn)

    def replace_all_repeated_elements(self, separator='.', mode='clone'):

        self._env_if_needed()
        env = self.env

        self.discard_tracker()
        unique_names = list(set(self.element_names))
        aux_dict = {nn: [] for nn in unique_names}
        for ii, nn in enumerate(self.element_names):
            aux_dict[nn].append(ii)

        for nn in unique_names:
            if len(aux_dict[nn]) > 1:
                i_rep = 0
                for ii in aux_dict[nn]:
                    while (new_name := nn + separator + str(i_rep)) in self.element_dict:
                        i_rep += 1
                    env.new(new_name, nn, mode=mode)
                    self.element_names[ii] = new_name

    def select(self, start=None, end=None, name=None):

        if start is xt.START:
            start = None

        if end is xt.END:
            end = None

        tt = self.get_table().rows[start:end]
        if tt.name[-1] == '_end_point':
            tt = tt.rows[:-1]

        self._env_if_needed()

        out = self.env.new_line(components=list(tt.env_name), name=name)
        out.particle_ref = self.particle_ref.copy() if self.particle_ref else None

        if hasattr(self, '_in_multiline') and self._in_multiline is not None:
            out.env._var_management = None
            out._var_management = None
            out.env._in_multiline = self._in_multiline
            out._in_multiline = self._in_multiline
            out.env._name_in_multiline = self._name_in_multiline
            out._name_in_multiline = self._name_in_multiline

        return out

    def set(self, name, *args, **kwargs):
        '''
        Set the values or expressions of variables or element properties.

        Parameters
        ----------
        name : str
            Name(s) of the variable or element.
        value: float or str
            Value or expression of the variable to set. Can be provided only
            if the name is associated to a variable.
        **kwargs, float or str
            Attributes to set. Can be provided only if the name is associated
            to an element.

        Examples
        --------
        >>> line.set('a', 0.1)
        >>> line.set('k1', '3*a')
        >>> line.set('quad', k1=0.1, k2='3*a')
        >>> line.set(['quad1', 'quad2'], k1=0.1, k2='3*a')
        >>> line.set(['c', 'd'], 0.1)
        >>> line.set(['e', 'f'], '3*a')

        '''
        if hasattr(name, 'env_name'):
            name = name.env_name
        elif hasattr(name, 'name'):
            name = name.name

        if isinstance(name, Iterable) and not isinstance(name, str):
            for nn in name:
                self.set(nn, *args, **kwargs)
            return

        _eval = self._xdeps_eval.eval

        if hasattr(self, 'lines') and name in self.lines:
            raise ValueError('Cannot set a line')

        if name in self.element_dict:
            if len(args) > 0:
                raise ValueError(f'Only kwargs are allowed when setting element attributes')

            extra = kwargs.pop('extra', None)

            ref_kwargs, value_kwargs = xt.environment._parse_kwargs(
                type(self.element_dict[name]), kwargs, _eval)
            xt.environment._set_kwargs(
                name=name, ref_kwargs=ref_kwargs, value_kwargs=value_kwargs,
                element_dict=self.element_dict, element_refs=self.element_refs)
            if extra is not None:
                assert isinstance(extra, dict), (
                    'Description must be a dictionary')
                if (not hasattr(self.element_dict[name], 'extra')
                    or not isinstance(self.element_dict[name].extra, dict)):
                    self.element_dict[name].extra = {}
                self.element_dict[name].extra.update(extra)
        else:
            if len(kwargs) > 0:
                raise ValueError(f'Only a single value is allowed when setting variable')
            if len(args) != 1:
                raise ValueError(f'A value must be provided when setting a variable')
            value = args[0]
            if 'extra' in kwargs and kwargs['extra'] is not None:
                raise ValueError(f'Extra is only allowed for elements')
            if isinstance(value, str):
                self.vars[name] = _eval(value)
            else:
                self.vars[name] = value

    def get(self, key):
        '''
        Get an element or the value of a variable.

        Parameters
        ----------
        key : str
            Name of the element or variable.

        Returns
        -------
        element : Element or float
            Element or value of the variable.

        '''

        if key in self.element_dict:
            return self.element_dict[key]
        elif key in self.vars:
            return self._xdeps_vref._owner[key]
        else:
            raise KeyError(f'Element or variable {key} not found')

    def info(self, key, limit=30):
        """
            Get information about an element or a variable.
        """

        if key in self.element_dict:
            return self[key].get_info()
        elif key in self.vars:
            return self.vars.info(key, limit=limit)
        else:
            raise KeyError(f'Element or variable {key} not found')

#    def get_value(self, key):
#        if key in self.element_dict:
#            return self.element_dict[key].get_value()
#        elif key in self.vars:
#            return self.vars.get_value(key)
#        else:
#            raise KeyError(f'Element or variable {key} not found')

    @property
    def ref_manager(self):
        return self._xdeps_manager

    def eval(self, expr):
        '''
        Get the value of an expression

        Parameters
        ----------
        expr : str
            Expression to evaluate.

        Returns
        -------
        value : float
            Value of the expression.
        '''

        return self.vars.eval(expr)

    def new_expr(self, expr):
        '''
        Create a new expression

        Parameters
        ----------
        expr : str
            Expression to create.

        Returns
        -------
        expr : Expression
            New expression.
        '''
        return self.vars.new_expr(expr)

    def get_expr(self, var):
        '''
        Get expression associated to a variable

        Parameters
        ----------
        var: str
            Name of the variable

        Returns
        -------
        expr : Expression
            Expression associated to the variable
        '''

        return self.vars.get_expr(var)

    def _env_if_needed(self):
        if not hasattr(self, 'env') or self.env is None:
            self.env = xt.Environment(element_dict=self.element_dict,
                                      particle_ref=self.particle_ref,
                                      _var_management=self._var_management)
            self.env._lines_weakrefs.add(self)

            # Temporary solution to keep consistency in multiline
            if hasattr(self, '_in_multiline') and self._in_multiline is not None:
                self.env._var_management = None
                self.env._in_multiline = self._in_multiline
                self.env._name_in_multiline = self._name_in_multiline

    def extend(self, line):
        self.element_names.extend(line.element_names)

    def __len__(self):
        return len(self.element_names)

    def items(self):
        for name in self.element_names:
            yield name, self.element_dict[name]

    def _var_management_to_dict(self):
        out = {}
        out['_var_management_data'] = deepcopy(self._var_management['data'])
        for kk in out['_var_management_data'].keys():
            if hasattr(out['_var_management_data'][kk], 'to_dict'):
                out['_var_management_data'][kk] = (
                    out['_var_management_data'][kk].to_dict())
        out['_var_manager'] = self._var_management['manager'].dump()
        return out

    def _has_valid_tracker(self):

        if self.tracker is None:
            return False
        try:
            self.tracker._check_invalidated()
            return True
        except:
            return False

    def _check_valid_tracker(self):
        if not self._has_valid_tracker():
            raise RuntimeError(
                "This line does not have a valid tracker. "
                "Please build the tracke using `line.build_tracker(...)`.")

    @property
    def name(self):
        '''Name of the line (if it is part of a `MultiLine`)'''
        if hasattr(self, '_in_multiline') and self._in_multiline is not None:
            for kk, vv in self._in_multiline.lines.items():
                if vv is self:
                    return kk
        else:
            return getattr(self, '_name', None)

    @property
    def iscollective(self):
        if not self._has_valid_tracker():
            raise RuntimeError(
                '`Line.iscollective` con only be called after `Line.build_tracker`')
        return self.tracker.iscollective

    @property
    def _buffer(self):
        if not self._has_valid_tracker():
            raise RuntimeError(
                '`Line._buffer` con only be called after `Line.build_tracker`')
        return self.tracker._buffer

    @property
    def _context(self):
        if not self._has_valid_tracker():
            raise RuntimeError(
                '`Line._context` con only be called after `Line.build_tracker`')
        return self.tracker._context

    def _init_var_management(self, dct=None):

        self._var_management = _make_var_management(element_dict=self.element_dict,
                                               dct=dct)

        if not hasattr(self, 'env') or self.env is None:
            self._env_if_needed()
            self.env._line_vars = LineVars(self.env)

    @property
    def _line_vars(self):
        return self.env._line_vars

    @property
    def record_last_track(self):
        self._check_valid_tracker()
        return self.tracker.record_last_track

    @property
    def vars(self):
        if hasattr(self, '_in_multiline') and self._in_multiline is not None:
            return self._in_multiline.vars
        else:
            return self._line_vars

    @property
    def varval(self):
        return self.vars.val

    @property
    def vv(self): # Shorter alias
        return self.vars.val

    @property
    def functions(self):
        return self._xdeps_fref

    @property
    def _xdeps_vref(self):
        if hasattr(self, '_in_multiline') and self._in_multiline is not None:
            return self._in_multiline._xdeps_vref
        if self._var_management is not None:
            return self._var_management['vref']

    @property
    def _xdeps_fref(self):
        if hasattr(self, '_in_multiline') and self._in_multiline is not None:
            return self._in_multiline._xdeps_fref
        if self._var_management is not None:
            return self._var_management['fref']

    @property
    def _xdeps_manager(self):
        if hasattr(self, '_in_multiline') and self._in_multiline is not None:
            return self._in_multiline._xdeps_manager
        if self._var_management is not None:
            return self._var_management['manager']

    @property
    def _xdeps_eval(self):
        try:
            eva_obj = self._xdeps_eval_obj
        except AttributeError:
            eva_obj = xd.madxutils.MadxEval(variables=self._xdeps_vref,
                                            functions=self._xdeps_fref,
                                            elements=self.element_dict,
                                            get='attr')
            self._xdeps_eval_obj = eva_obj

        return eva_obj

    @property
    def element_refs(self):
        if hasattr(self, '_in_multiline'):
            var_sharing = self._in_multiline._var_sharing
            if var_sharing is not None:
                return var_sharing._eref[self._name_in_multiline]
        if self._var_management is not None:
            return self._var_management['lref']

    @property
    def element_dict(self):
        return self._element_dict

    @element_dict.setter
    def element_dict(self, value):
        if self._element_dict is None:
            self._element_dict = {}
        self._element_dict.clear()
        self._element_dict.update(value)

    @property
    def element_names(self):
        return self._element_names

    @element_names.setter
    def element_names(self, value):
        if not hasattr(self, '_element_names'):
            self._element_names = []
        self._frozen_check()
        self._element_names = value

    @property
    def elements(self):
        return tuple([self.element_dict[nn] for nn in self.element_names])

    @property
    def skip_end_turn_actions(self):
        return self._extra_config['skip_end_turn_actions']

    @skip_end_turn_actions.setter
    def skip_end_turn_actions(self, value):
        self._extra_config['skip_end_turn_actions'] = value

    @property
    def reset_s_at_end_turn(self):
        return self._extra_config['reset_s_at_end_turn']

    @reset_s_at_end_turn.setter
    def reset_s_at_end_turn(self, value):
        self._extra_config['reset_s_at_end_turn'] = value

    @property
    def matrix_responsiveness_tol(self):
        return self._extra_config['matrix_responsiveness_tol']

    @matrix_responsiveness_tol.setter
    def matrix_responsiveness_tol(self, value):
        self._extra_config['matrix_responsiveness_tol'] = value

    @property
    def matrix_stability_tol(self):
        return self._extra_config['matrix_stability_tol']

    @matrix_stability_tol.setter
    def matrix_stability_tol(self, value):
        self._extra_config['matrix_stability_tol'] = value

    @property
    def _radiation_model(self):
        return self._extra_config['_radiation_model']

    @_radiation_model.setter
    def _radiation_model(self, value):
        self._extra_config['_radiation_model'] = value

    @property
    def _spin_model(self):
        return self._extra_config['_spin_model']

    @_spin_model.setter
    def _spin_model(self, value):
        self._extra_config['_spin_model'] = value

    @property
    def _beamstrahlung_model(self):
        return self._extra_config['_beamstrahlung_model']

    @_beamstrahlung_model.setter
    def _beamstrahlung_model(self, value):
        self._extra_config['_beamstrahlung_model'] = value

    @property
    def _bhabha_model(self):
        return self._extra_config['_bhabha_model']

    @_bhabha_model.setter
    def _bhabha_model(self, value):
        self._extra_config['_bhabha_model'] = value

    @property
    def _needs_rng(self):
        return self._extra_config['_needs_rng']

    @_needs_rng.setter
    def _needs_rng(self, value):
        self._extra_config['_needs_rng'] = value

    @property
    def enable_time_dependent_vars(self):
        return self._extra_config['enable_time_dependent_vars']

    @enable_time_dependent_vars.setter
    def enable_time_dependent_vars(self, value):
        assert value in (True, False)
        self._extra_config['enable_time_dependent_vars'] = value

    @property
    def dt_update_time_dependent_vars(self):
        return self._extra_config['dt_update_time_dependent_vars']

    @dt_update_time_dependent_vars.setter
    def dt_update_time_dependent_vars(self, value):
        self._extra_config['dt_update_time_dependent_vars'] = value

    @property
    def _t_last_update_time_dependent_vars(self):
        return self._extra_config['_t_last_update_time_dependent_vars']

    @_t_last_update_time_dependent_vars.setter
    def _t_last_update_time_dependent_vars(self, value):
        self._extra_config['_t_last_update_time_dependent_vars'] = value

    @property
    def time_last_track(self):
        self._check_valid_tracker()
        return self.tracker.time_last_track

    @property
    def twiss_default(self):
        return self._extra_config['twiss_default']

    @property
    def energy_program(self):
        try:
            out = self.element_dict['energy_program']
        except KeyError:
            out = None
        return out

    @energy_program.setter
    def energy_program(self, value):
        if value is None:
            if 'energy_program' in self.element_dict:
                del self.element_dict['energy_program']
            return
        self.element_dict['energy_program'] = value
        assert self.vars is not None, (
            'Xdeps expression need to be enabled to use `energy_program`')
        if self.energy_program.needs_complete:
            self.energy_program.complete_init(self)
        self.energy_program.line = self
        self.element_refs['energy_program'].t_turn_s_line = self.vars['t_turn_s']

    @property
    def steering_monitors_x(self):
        return self._extra_config.get('steering_monitors_x', None)

    @steering_monitors_x.setter
    def steering_monitors_x(self, value):
        self._extra_config['steering_monitors_x'] = value

    @property
    def steering_monitors_y(self):
        return self._extra_config.get('steering_monitors_y', None)

    @steering_monitors_y.setter
    def steering_monitors_y(self, value):
        self._extra_config['steering_monitors_y'] = value

    @property
    def steering_correctors_x(self):
        return self._extra_config.get('steering_correctors_x', None)

    @steering_correctors_x.setter
    def steering_correctors_x(self, value):
        self._extra_config['steering_correctors_x'] = value

    @property
    def steering_correctors_y(self):
        return self._extra_config.get('steering_correctors_y', None)

    @steering_correctors_y.setter
    def steering_correctors_y(self, value):
        self._extra_config['steering_correctors_y'] = value

    @property
    def corrector_limits_x(self):
        return self._extra_config.get('corrector_limits_x', None)

    @corrector_limits_x.setter
    def corrector_limits_x(self, value):
        self._extra_config['corrector_limits_x'] = value

    @property
    def corrector_limits_y(self):
        return self._extra_config.get('corrector_limits_y', None)

    @corrector_limits_y.setter
    def corrector_limits_y(self, value):
        self._extra_config['corrector_limits_y'] = value

    def __getitem__(self, key):
        if np.issubdtype(key.__class__, np.integer):
            key = self.element_names[key]
        assert isinstance(key, str)
        if key in self.element_dict:
            if self.element_refs is None:
                return self.element_dict[key]
            return xd.madxutils.View(
                self.element_dict[key], self.element_refs[key],
                evaluator=self._xdeps_eval.eval)
        elif key in self.vars:
            return self.vv[key]
        elif hasattr(self, 'lines') and key in self.lines: # Want to reuse the method for the env
            return self.lines[key]
        elif "::" in key and (env_name := key.split("::")[0]) in self.element_dict:
            return self[env_name]
        else:
            raise KeyError(f'Name {key} not found')

    def __setitem__(self, key, value):

        if isinstance(value, Line):
            raise ValueError('Cannot set a Line, please use Envirnoment.new_line')
            # Would need to make sure they refer to the same environment

        if np.isscalar(value) or xd.refs.is_ref(value):
            if key in self.element_dict:
                raise ValueError(f'There is already an element with name {key}')
            self.vars[key] = value
        else:
            raise ValueError('Only scalars or references are allowed')

    def _get_non_collective_line(self):
        if not self.iscollective:
            return self
        else:
            # Shallow copy of the line
            out = Line.__new__(Line)
            out.__dict__.update(self.__dict__)

            # Change the element dict (beware of the element_dict property)
            out._element_dict = self.tracker._element_dict_non_collective

            # Shallow copy of the tracker
            out.tracker = self.tracker.__new__(self.tracker.__class__)
            out.tracker.__dict__.update(self.tracker.__dict__)
            out.tracker.iscollective = False
            out.tracker.line = out

            return out

    def _get_attr_cache(self):
        cache = LineAttr(
            line=self,
            fields={
                'delta_taper': None, 'ks': None,
                'voltage': None, 'frequency': None, 'lag': None,
                'lag_taper': None,

                'weight': None,

                '_own_length': 'length',

                '_own_sin_rot_s': '_sin_rot_s',
                '_own_cos_rot_s': '_cos_rot_s',
                '_own_shift_x': '_shift_x',
                '_own_shift_y': '_shift_y',
                '_own_shift_s': '_shift_s',

                '_own_h': 'h',
                '_own_hxl': 'hxl',

                '_own_radiation_flag': 'radiation_flag',

                '_own_k0': 'k0',
                '_own_k1': 'k1',
                '_own_k2': 'k2',
                '_own_k3': 'k3',
                '_own_k4': 'k4',
                '_own_k5': 'k5',

                '_own_k0s': 'k0s',
                '_own_k1s': 'k1s',
                '_own_k2s': 'k2s',
                '_own_k3s': 'k3s',
                '_own_k4s': 'k4s',
                '_own_k5s': 'k5s',

                '_own_k0l': ('knl', 0),
                '_own_k1l': ('knl', 1),
                '_own_k2l': ('knl', 2),
                '_own_k3l': ('knl', 3),
                '_own_k4l': ('knl', 4),
                '_own_k5l': ('knl', 5),

                '_own_k0sl': ('ksl', 0),
                '_own_k1sl': ('ksl', 1),
                '_own_k2sl': ('ksl', 2),
                '_own_k3sl': ('ksl', 3),
                '_own_k4sl': ('ksl', 4),
                '_own_k5sl': ('ksl', 5),

                # Handling of reference frame transformations
                # (XYShift, XRotation, YRotation, SRotation)
                # TODO: The dx, dy, etc labels come from the element level and should possibly be changed
                '_own_ref_shift_x':         'dx',
                '_own_ref_shift_y':         'dy',
                '_own_ref_rot_sin_angle':   'sin_angle',
                '_own_ref_rot_cos_angle':   'cos_angle',
                '_own_ref_rot_sin_z':       'sin_z',
                '_own_ref_rot_cos_z':       'cos_z',

                '_parent_length': (('_parent', 'length'), None),
                '_parent_sin_rot_s': (('_parent', '_sin_rot_s'), None),
                '_parent_cos_rot_s': (('_parent', '_cos_rot_s'), None),
                '_parent_shift_x': (('_parent', '_shift_x'), None),
                '_parent_shift_y': (('_parent', '_shift_y'), None),
                '_parent_shift_s': (('_parent', '_shift_s'), None),

                '_parent_h': (('_parent', 'h'), None),
                '_parent_hxl': (('_parent', 'hxl'), None),
                '_parent_rbend_model': (('_parent', 'rbend_model'), None),

                '_parent_radiation_flag': (('_parent', 'radiation_flag'), None),

                '_parent_k0': (('_parent', 'k0'), None),
                '_parent_k1': (('_parent', 'k1'), None),
                '_parent_k2': (('_parent', 'k2'), None),
                '_parent_k3': (('_parent', 'k3'), None),
                '_parent_k4': (('_parent', 'k4'), None),
                '_parent_k5': (('_parent', 'k5'), None),

                '_parent_k0s': (('_parent', 'k0s'), None),
                '_parent_k1s': (('_parent', 'k1s'), None),
                '_parent_k2s': (('_parent', 'k2s'), None),
                '_parent_k3s': (('_parent', 'k3s'), None),
                '_parent_k4s': (('_parent', 'k4s'), None),
                '_parent_k5s': (('_parent', 'k5s'), None),

                '_parent_k0l': (('_parent', 'knl'), 0),
                '_parent_k1l': (('_parent', 'knl'), 1),
                '_parent_k2l': (('_parent', 'knl'), 2),
                '_parent_k3l': (('_parent', 'knl'), 3),
                '_parent_k4l': (('_parent', 'knl'), 4),
                '_parent_k5l': (('_parent', 'knl'), 5),

                '_parent_k0sl': (('_parent', 'ksl'), 0),
                '_parent_k1sl': (('_parent', 'ksl'), 1),
                '_parent_k2sl': (('_parent', 'ksl'), 2),
                '_parent_k3sl': (('_parent', 'ksl'), 3),
                '_parent_k4sl': (('_parent', 'ksl'), 4),
                '_parent_k5sl': (('_parent', 'ksl'), 5),

                # Handling of reference frame transformations
                # (XYShift, XRotation, YRotation, SRotation)
                # TODO: The dx, dy, etc labels come from the element level and should possibly be changed
                '_parent_ref_shift_x': (('_parent', 'dx'), None),
                '_parent_ref_shift_y': (('_parent', 'dy'), None),
                '_parent_ref_rot_sin_angle': (('_parent', 'sin_angle'), None),
                '_parent_ref_rot_cos_angle': (('_parent', 'cos_angle'), None),
                '_parent_ref_rot_sin_z': (('_parent', 'sin_z'), None),
                '_parent_ref_rot_cos_z': (('_parent', 'cos_z'), None),

            },
            derived_fields={
                'length': lambda attr:
                    attr['_own_length'] + attr['_parent_length'] * attr['weight'],
                '_angle_force_body': _angle_force_body_from_attr,
                'angle_rad': _angle_rbend_correction_from_attr,
                'rot_s_rad': _rot_s_from_attr,
                'shift_x': lambda attr:
                    attr['_own_shift_x'] + attr['_parent_shift_x']
                    * attr._rot_and_shift_from_parent,
                'shift_y': lambda attr:
                    attr['_own_shift_y'] + attr['_parent_shift_y']
                    * attr._rot_and_shift_from_parent,
                'shift_s': lambda attr:
                    attr['_own_shift_s'] + attr['_parent_shift_s']
                    * attr._rot_and_shift_from_parent,
                'radiation_flag': lambda attr:
                    attr['_own_radiation_flag'] * (attr['_own_radiation_flag'] != ID_RADIATION_FROM_PARENT)
                  + attr['_parent_radiation_flag'] * (attr['_own_radiation_flag'] == ID_RADIATION_FROM_PARENT),
                'k0l': lambda attr: (
                    attr['_own_k0l']
                    + attr['_own_k0'] * attr['_own_length']
                    + attr['_parent_k0l'] * attr['weight'] * attr._inherit_strengths
                    + attr['_parent_k0'] * attr['_parent_length'] * attr['weight'] * attr._inherit_strengths),
                'k0sl': lambda attr: (
                    attr['_own_k0sl']
                    + attr['_own_k0s'] * attr['_own_length']
                    + attr['_parent_k0sl'] * attr['weight']* attr._inherit_strengths
                    + attr['_parent_k0s'] * attr['_parent_length'] * attr['weight'] * attr._inherit_strengths),
                'k1l': lambda attr: (
                    attr['_own_k1l']
                    + attr['_own_k1'] * attr['_own_length']
                    + attr['_parent_k1l'] * attr['weight'] * attr._inherit_strengths
                    + attr['_parent_k1'] * attr['_parent_length'] * attr['weight']* attr._inherit_strengths),
                'k1sl': lambda attr: (
                    attr['_own_k1sl']
                    + attr['_own_k1s'] * attr['_own_length']
                    + attr['_parent_k1sl'] * attr['weight'] * attr._inherit_strengths
                    + attr['_parent_k1s'] * attr['_parent_length'] * attr['weight'] * attr._inherit_strengths),
                'k2l': lambda attr: (
                    attr['_own_k2l']
                    + attr['_own_k2'] * attr['_own_length']
                    + attr['_parent_k2l'] * attr['weight'] * attr._inherit_strengths
                    + attr['_parent_k2'] * attr['_parent_length'] * attr['weight'] * attr._inherit_strengths),
                'k2sl': lambda attr: (
                    attr['_own_k2sl']
                    + attr['_own_k2s'] * attr['_own_length']
                    + attr['_parent_k2sl'] * attr['weight'] * attr._inherit_strengths
                    + attr['_parent_k2s'] * attr['_parent_length'] * attr['weight'] * attr._inherit_strengths),
                'k3l': lambda attr: (
                    attr['_own_k3l']
                    + attr['_own_k3'] * attr['_own_length']
                    + attr['_parent_k3l'] * attr['weight'] * attr._inherit_strengths
                    + attr['_parent_k3'] * attr['_parent_length'] * attr['weight'] * attr._inherit_strengths),
                'k3sl': lambda attr: (
                    attr['_own_k3sl']
                    + attr['_own_k3s'] * attr['_own_length']
                    + attr['_parent_k3sl'] * attr['weight'] * attr._inherit_strengths
                    + attr['_parent_k3s'] * attr['_parent_length'] * attr['weight'] * attr._inherit_strengths),
                'k4l': lambda attr: (
                    attr['_own_k4l']
                    + attr['_own_k4'] * attr['_own_length']
                    + attr['_parent_k4l'] * attr['weight'] * attr._inherit_strengths
                    + attr['_parent_k4'] * attr['_parent_length'] * attr['weight'] * attr._inherit_strengths),
                'k4sl': lambda attr: (
                    attr['_own_k4sl']
                    + attr['_own_k4s'] * attr['_own_length']
                    + attr['_parent_k4sl'] * attr['weight'] * attr._inherit_strengths
                    + attr['_parent_k4s'] * attr['_parent_length'] * attr['weight'] * attr._inherit_strengths),
                'k5l': lambda attr: (
                    attr['_own_k5l']
                    + attr['_own_k5'] * attr['_own_length']
                    + attr['_parent_k5l'] * attr['weight'] * attr._inherit_strengths
                    + attr['_parent_k5'] * attr['_parent_length'] * attr['weight'] * attr._inherit_strengths),
                'k5sl': lambda attr: (
                    attr['_own_k5sl']
                    + attr['_own_k5s'] * attr['_own_length']
                    + attr['_parent_k5sl'] * attr['weight'] * attr._inherit_strengths
                    + attr['_parent_k5s'] * attr['_parent_length'] * attr['weight'] * attr._inherit_strengths),
                'hkick': lambda attr: attr["angle_rad"] - attr["k0l"],
                'vkick': lambda attr: attr["k0sl"],
                'ref_shift_x': lambda attr: attr['_own_ref_shift_x'] + attr['_parent_ref_shift_x'],
                'ref_shift_y': lambda attr: attr['_own_ref_shift_y'] + attr['_parent_ref_shift_y'],
                'ref_rot_angle_rad': lambda attr: np.arctan2(
                    attr['_own_ref_rot_sin_angle'] + attr['_parent_ref_rot_sin_angle'] +\
                    attr['_own_ref_rot_sin_z'] + attr['_parent_ref_rot_sin_z'],
                    attr['_own_ref_rot_cos_angle'] + attr['_parent_ref_rot_cos_angle'] +\
                    attr['_own_ref_rot_cos_z'] + attr['_parent_ref_rot_cos_z']),
            }
        )
        return cache

    def _insert_thin_elements_at_s(self, elements_to_insert, s_tol=0.5e-6):

        '''
        Example:
        elements_to_insert = [
            # s .    # elements to insert (name, element)
            (s0,     [(f'm0_at_a', xt.Marker()), (f'm1_at_a', xt.Marker()), (f'm2_at_a', xt.Marker())]),
            (s0+10., [(f'm0_at_b', xt.Marker()), (f'm1_at_b', xt.Marker()), (f'm2_at_b', xt.Marker())]),
            (s1,     [(f'm0_at_c', xt.Marker()), (f'm1_at_c', xt.Marker()), (f'm2_at_c', xt.Marker())]),
        ]

        '''

        self._frozen_check()

        s_cuts = [ee[0] for ee in elements_to_insert]
        s_cuts = np.sort(s_cuts)

        self.cut_at_s(s_cuts, s_tol=s_tol)

        tt_after_cut = self.get_table()

        # Names for insertions
        ele_name_insertions = []
        for s_insert, ee in progress(elements_to_insert, desc="Locate insertion points"):
            # Find element_name for insertion
            ii_ins = np.where(tt_after_cut['s'] >= s_insert - s_tol)[0][0]
            ele_name_insertions.append(tt_after_cut['name'][ii_ins])
            assert np.abs(s_insert - tt_after_cut['s'][ii_ins]) < s_tol

        # Add all elements to self.element_dict
        for s_insert, ee in elements_to_insert:
            for nn, el in ee:
                assert nn not in self.element_dict
                self.element_dict[nn] = el

        # Insert elements
        for i_ins, (s_insert, ee) in enumerate(
                    progress(elements_to_insert, desc="Inserting elements")):
            ele_name_ins = ele_name_insertions[i_ins]

            if ele_name_ins not in self.element_names:
                assert ele_name_ins == '_end_point'
                insert_at = None
            else:
                insert_at = self.element_names.index(ele_name_ins)
            for nn, el in ee:

                assert el.isthick == False
                if insert_at is None:
                    self.element_names.append(nn)
                else:
                    self.element_names.insert(insert_at, nn)

                if insert_at is not None:
                    insert_at += 1

    def _insert_thick_elements_at_s(self, element_names, elements,
                                    at_s, s_tol=1e-6):

        assert isinstance(element_names, (list, tuple))
        assert isinstance(elements, (list, tuple))
        assert isinstance(at_s, (list, tuple, np.ndarray))
        assert len(element_names) == len(elements) == len(at_s)

        self._frozen_check()

        s_insert = np.array(at_s)
        l_insert = np.array([_length(ee, None) for ee in elements])
        ele_insert = list(elements).copy()
        name_insert = list(element_names).copy()

        end_insert = np.array(s_insert) + np.array(l_insert)

        self.cut_at_s(list(s_insert) + list(end_insert))

        i_sorted = np.argsort(s_insert)
        s_insert_sorted = s_insert[i_sorted]
        ele_insert_sorted = [ele_insert[i] for i in i_sorted]
        name_insert_sorted = [name_insert[i] for i in i_sorted]
        end_insert_sorted = end_insert[i_sorted]

        assert np.all(s_insert_sorted[:-1] < end_insert_sorted[1:]), (
                    'Overlapping insertions')

        old_element_names = self.element_names

        s_tol = 1e-6

        s_vect_upstream = np.array(self.get_s_position(mode='upstream'))

        i_replace = np.zeros(len(s_vect_upstream), dtype=int)
        mask_remove = np.zeros(len(s_vect_upstream), dtype=bool)

        i_replace[:] = -1

        for ii in range(len(s_insert_sorted)):
            ss_start = s_insert_sorted[ii]
            ss_end = end_insert_sorted[ii]

            i_first_removal = np.where(np.abs(s_vect_upstream - ss_start) < s_tol)[0][-1]
            i_last_removal = np.where(np.abs(s_vect_upstream - ss_end) < s_tol)[0][0] - 1

            i_replace[i_first_removal] = ii
            mask_remove[i_first_removal+1:i_last_removal+1] = True

        new_element_names = []
        for ii, nn in enumerate(old_element_names):
            if mask_remove[ii]:
                continue
            if i_replace[ii] != -1:
                new_element_names.append(name_insert_sorted[i_replace[ii]])
            else:
                new_element_names.append(nn)

        for new_nn, new_ee in zip(name_insert_sorted, ele_insert_sorted):
            self.element_dict[new_nn] = new_ee

        self.element_names = new_element_names

    @property
    def _line_before_slicing(self):
        if self._element_names_before_slicing is None:
            return None

        if self._line_before_slicing_cache is None:
            # Shallow copy of the line
            out = Line.__new__(Line)
            out.__dict__.update(self.__dict__)
            out._element_names = self._element_names_before_slicing
            out.tracker = None
            self._line_before_slicing_cache = out

        return self._line_before_slicing_cache

    def _replace_with_equivalent_elements(self):

        self._frozen_check()

        for nn in self.element_names:
            ee = self.element_dict[nn]
            if hasattr(ee, 'get_equivalent_element'):
                new_ee = ee.get_equivalent_element()
                self.element_dict[nn] = new_ee

    @property
    def _element_names_unique(self):
        if not self._has_valid_tracker():
            raise RuntimeError(
                '`Line._element_names_unique` con only be called after `Line.build_tracker`')
        return self.tracker._tracker_data_base._element_names_unique

def frac(x):
    return x % 1

def sinc(x):
    return np.sinc(x / np.pi)

class Functions:

    _mathfunctions = dict(
        sqrt = math.sqrt,
        log = math.log,
        log10 = math.log10,
        exp = math.exp,
        sin = math.sin,
        cos = math.cos,
        tan = math.tan,
        asin = math.asin,
        acos = math.acos,
        atan = math.atan,
        atan2 = math.atan2,
        sinh = math.sinh,
        cosh = math.cosh,
        tanh = math.tanh,
        sinc = sinc,
        abs = math.fabs,
        erf = math.erf,
        erfc = math.erfc,
        floor = math.floor,
        ceil = math.ceil,
        round = np.round,
        frac = frac,
    )

    def __init__(self):
        object.__setattr__(self, '_funcs', {})

    def __setitem__(self, name, value):
        self._funcs[name] = value

    def __getitem__(self, name):
        if name in self._funcs:
            return self._funcs[name]
        elif name in self._mathfunctions:
            return self._mathfunctions[name]
        else:
            raise KeyError(f'Unknown function {name}')

    def __getattr__(self, name):
        if name == '_funcs':
            return object.__getattribute__(self, '_funcs')
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f'Unknown function {name}')

    def update(self, other):
        self._funcs.update(other._funcs)

    def to_dict(self):
        fdict = {}
        for kk, ff in self._funcs.items():
            fdict[kk] = ff.to_dict()
            fdict[kk]['__class__'] = ff.__class__.__name__
        out = {'_funcs': fdict}
        return out

    @classmethod
    def from_dict(cls, dct):
        _funcs = {}
        for kk, ff in dct['_funcs'].items():
            ffcls = getattr(xd, ff.pop('__class__'))
            _funcs[kk] = ffcls.from_dict(ff)
        out = cls()
        out._funcs.update(_funcs)
        return out


def _deserialize_element(el, class_dict, _buffer):
    eldct = el.copy()
    eltype = class_dict[eldct.pop('__class__')]
    if hasattr(eltype, '_XoStruct'):
        return eltype.from_dict(eldct, _buffer=_buffer)
    else:
        return eltype.from_dict(eldct)

def _is_simple_quadrupole(el):
    if not isinstance(el, Multipole):
        return False
    return (el.radiation_flag == 0
            and (el.order == 1 or len(el.knl) == 2 or not any(el.knl[2:]))
            and el.knl[0] == 0
            and not any(el.ksl)
            and not el.hxl
            and el.shift_x == 0 and el.shift_y == 0 and el.shift_s == 0
            and np.abs(el.rot_s_rad) < 1e-12)

def _is_simple_dipole(el):
    if not isinstance(el, Multipole):
        return False
    return (el.radiation_flag == 0
            and (el.order == 0 or len(el.knl) == 1 or not any(el.knl[1:]))
            and not any(el.ksl)
            and el.shift_x == 0 and el.shift_y == 0 and el.shift_s == 0
            and np.abs(el.rot_s_rad) < 1e-12)

@contextmanager
def freeze_longitudinal(tracker):
    """Context manager to freeze longitudinal motion in a tracker."""
    from xtrack.tracker import TrackerConfig
    config = TrackerConfig()
    config.update(tracker.config)
    tracker.freeze_longitudinal(True)
    try:
        yield None
    finally:
        tracker.config.clear()
        tracker.config.update(config)


_freeze_longitudinal = freeze_longitudinal  # to avoid name clash with function argument


def mk_class_namespace(extra_classes):
    try:
        import xfields as xf
        all_classes = element_classes + xf.element_classes + extra_classes + (Line,)
    except ImportError:
        all_classes = element_classes + extra_classes
        log.warning("Xfields not installed")
    try:
        import xcoll as xc
        all_classes += xc.element_classes
    except ImportError:
        log.warning("Xcoll not installed")

    all_classes = all_classes + (EnergyProgram, xt.Replica)

    out = AttrDict()
    for cl in all_classes:
        out[cl.__name__] = cl
    return out

def _length(element, line):
    if isinstance(element, xt.Replica):
        element = element.resolve(line)
    if hasattr(element, 'length'):
        return element.length
    assert hasattr(element, 'parent_name')
    return line.element_dict[element.parent_name].length * element.weight

def _is_drift(element, line):
    if isinstance(element, xt.Replica):
        element = element.resolve(line)
    if isinstance(element, beam_elements.Drift):
        return True
    if element.__class__.__name__.startswith('Drift'):
        return True
    return False

def _behaves_like_drift(element, line):
    if _is_drift(element, line):
        return True
    if isinstance(element, xt.Replica):
        element = element.resolve(line)
    return hasattr(element, 'behaves_like_drift') and element.behaves_like_drift

def _is_aperture(element, line):
    if isinstance(element, xt.Replica):
        element = element.resolve(line)
    return element.__class__.__name__.startswith('Limit')

def _is_thick(element, line):
    if isinstance(element, xt.Replica):
        element = element.resolve(line)
    return hasattr(element, "isthick") and element.isthick

def _is_collective(element, line):
    if isinstance(element, xt.Replica):
        element = element.resolve(line)
    iscoll = not hasattr(element, 'iscollective') or element.iscollective
    return iscoll

# whether backtrack in loss location refinement is allowed
def _allow_loss_refinement(element, line):
    if isinstance(element, xt.Replica):
        element = element.resolve(line)
    return hasattr(element, 'allow_loss_refinement') and element.allow_loss_refinement

# whether element has backtrack capability
def _has_backtrack(element, line):
    if isinstance(element, xt.Replica):
        element = element.resolve(line)
    return hasattr(element, 'has_backtrack') and element.has_backtrack

def _next_name(prefix, names, name_format='{}{}'):
    """Return an available element name by appending a number"""
    if prefix not in names: return prefix
    i = 1
    while name_format.format(prefix, i) in names:
        i += 1
    return name_format.format(prefix, i)

def _dicts_equal(dict1, dict2):
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return False
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    for key in dict1.keys():
        if hasattr(dict1[key], '__iter__'):
            if not hasattr(dict2[key], '__iter__'):
                return False
            elif isinstance(dict1[key], dict):
                if not isinstance(dict2[key], dict):
                    return False
                else:
                    if not _dicts_equal(dict1[key], dict2[key]):
                        return False
            elif not np.array_equal(dict1[key], dict2[key]):
                return False
        elif dict1[key] != dict2[key]:
            return False
    return True

def _apertures_equal(ap1, ap2, line):
    if not _is_aperture(ap1, line) or not _is_aperture(ap2, line):
        raise ValueError(f"Element {ap1} or {ap2} not an aperture!")
    if isinstance(ap1, xt.Replica):
        ap1 = ap1.resolve(line)
    if isinstance(ap2, xt.Replica):
        ap2 = ap2.resolve(line)
    if ap1.__class__ != ap2.__class__:
        return False
    ap1 = ap1.to_dict()
    ap2 = ap2.to_dict()
    return _dicts_equal(ap1, ap2)


def _lines_equal(line1, line2):
    d1 = line1.to_dict()
    d2 = line2.to_dict()
    d1.pop('_var_management_data', None)
    d2.pop('_var_management_data', None)
    d1.pop('_var_manager', None)
    d2.pop('_var_manager', None)
    out = _dicts_equal(d1, d2)
    return out


DEG2RAD = np.pi / 180.


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Node:
    def __init__(self, s, what, *, from_=0, name=None):
        """Holds the location of an element or sequence for use with Line.from_sequence

        Args:
            s (float): Location (in m) of what relative to from_.
            what (str, BeamElement or list): Object to place here. Can be an instance of a BeamElement,
                another sequence given as list of At, or the name of a named element.
            from_ (float or str, optional): Reference location for placement, can be the s coordinate (in m)
                or the name of an element or sequence whose location is used.
            name (str, optional): Name of the element to place here. If None, a name is chosen automatically.

        """
        self.s = s
        self.from_ = from_
        self.what = what
        self.name = name

    def __repr__(self):
        return f"Node({self.s}, {self.what}, from_={self.from_}, name={self.name})"


At = Node


def flatten_sequence(nodes, elements={}, sequences={}, copy_elements=False, naming_scheme='{}{}'):
    """Flatten the sequence definition

    Named elements and nested sequences are replaced recursively.
    Node locations are made absolute.

    See Line.from_sequence for details
    """
    flat_nodes = []
    for node in nodes:
        # determine absolute position
        s = node.s
        if isinstance(node.from_, str):
            # relative to another element
            for n in flat_nodes:
                if node.from_ == n.name:
                    s += n.s
                    break
            else:
                raise ValueError(f'Unknown element name {node.from_} passed as from_')
        else:
            s += node.from_

        # find a unique name
        name = node.name or (node.what if isinstance(node.what, str) else 'element')
        name = _next_name(name, [n.name for n in flat_nodes], naming_scheme)

        # determine what to place here
        element = None
        sequence = None
        if isinstance(node.what, str):
            if node.what in elements:
                element = elements[node.what]
                if copy_elements:
                    element = element.copy()
            elif node.what in sequences:
                sequence = sequences[node.what]
            else:
                raise ValueError(f'Unknown element or sequence name {node.what}')
        elif isinstance(node.what, BeamElement):
            element = node.what
        elif hasattr(node.what, '__iter__'):
            sequence = node.what
        else:
            raise ValueError(f'Unknown element type {node.what}')

        # place elements
        if element is not None:
            flat_nodes.append(Node(s, element, name=name))

        # place nested sequences by recursion
        if sequence is not None:
            flat_nodes.append(Node(s, Marker(), name=name))
            for sub in flatten_sequence(sequence, elements=elements, sequences=sequences, copy_elements=copy_elements, naming_scheme=naming_scheme):
                sub_name = naming_scheme.format(name, sub.name)
                flat_nodes.append(Node(s + sub.s, sub.what, name=sub_name))

    return flat_nodes


@contextmanager
def _preserve_config(ln_or_trk):
    from xtrack.tracker import TrackerConfig
    config = TrackerConfig()
    config.update(ln_or_trk.config)
    try:
        yield
    finally:
        ln_or_trk.config.clear()
        ln_or_trk.config.update(config)


@contextmanager
def freeze_longitudinal(ln_or_trk):
    """Context manager to freeze longitudinal motion in a tracker."""
    from xtrack.tracker import TrackerConfig
    config = TrackerConfig()
    config.update(ln_or_trk.config)
    ln_or_trk.freeze_longitudinal(True)
    try:
        yield None
    finally:
        ln_or_trk.config.clear()
        ln_or_trk.config.update(config)


@contextmanager
def _temp_knobs(line_or_trk, knobs: dict):
    old_values = {kk: line_or_trk.vars[kk]._value for kk in knobs.keys()}
    try:
        for kk, vv in knobs.items():
            line_or_trk.vars[kk] = vv
        yield
    finally:
        for kk, vv in old_values.items():
            line_or_trk.vars[kk] = vv

class _DefaultFactory:
    def __init__(self, default):
        self.default = default

    def __call__(self):
        return self.default

class LineVars:

    def __init__(self, line):
        self.line = line
        if '__vary_default' not in self.line._xdeps_vref._owner.keys():
            self.line._xdeps_vref._owner['__vary_default'] = {}
        self.val = VarValues(self)
        self.vars_to_update = WeakSet()

    def keys(self):
        if self.line._xdeps_vref is None:
            raise RuntimeError(
                f'Cannot access variables as the line has no xdeps manager')
        out = list(self.line._xdeps_vref._owner.keys()).copy()
        return out

    def __iter__(self):
        raise NotImplementedError('Use keys() method') # Untested
        return self.line._xdeps_vref._owner.__iter__()

    def update(self, *args, **kwargs):
        default_to_zero = kwargs.pop('default_to_zero', None)
        old_default_to_zero = self.default_to_zero
        if default_to_zero is not None:
            self.default_to_zero = default_to_zero
        try:
            if self.line._xdeps_vref is None:
                raise RuntimeError(
                    f'Cannot access variables as the line has no xdeps manager')
            if len(args) > 0:
                assert len(args) == 1, 'update expected at most 1 positional argument'
                other = args[0]
                for kk in other.keys():
                    self[kk] = other[kk]
            for kk, vv in kwargs.items():
                self[kk] = vv
        except Exception as ee:
            if default_to_zero is not None:
                self.default_to_zero = old_default_to_zero
            raise ee
        if default_to_zero is not None:
            self.default_to_zero = old_default_to_zero

    def load(
            self,
            file=None,
            string=None,
            format: Literal['json', 'madx', 'python'] = None,
            timeout=5.,
        ):

        if isinstance(file, Path):
            file = str(file)

        if (file is None) == (string is None):
            raise ValueError('Must specify either file or string, but not both')

        FORMATS = {'json', 'madx', 'python'}
        if string and format not in FORMATS:
            raise ValueError(f'Format must be specified to be one of {FORMATS} when '
                            f'using string input')

        if format is None and file is not None:
            if file.endswith('.json') or file.endswith('.json.gz'):
                format = 'json'
            elif file.endswith('.str') or file.endswith('.madx'):
                format = 'madx'
            elif file.endswith('.py'):
                format = 'python'

        if file and (file.startswith('http://') or file.startswith('https://')):
            string = xt.general.read_url(file, timeout=timeout)
            file = None

        if format == 'json':
            ddd = xt.json.load(file=file, string=string)
            self.update(ddd, default_to_zero=True)
        elif format == 'madx':
            return self.load_madx(file, string)
        elif format == 'python':
            if string is not None:
                raise NotImplementedError('Loading from string not implemented for python format')
            env = xt.Environment()
            env.call(file)
            self.update(env.vars.get_table().to_dict(), default_to_zero=True)
            return env

    @property
    def vary_default(self):
        if self.line._xdeps_vref is None:
            raise RuntimeError(
                f'Cannot access variables as the line has no xdeps manager')
        return self.line._xdeps_vref._owner['__vary_default']

    def get_table(self, compact=True):
        if self.line._xdeps_vref is None:
            raise RuntimeError(
                f'Cannot access variables as the line has no xdeps manager')
        name = np.array([kk for kk in list(self.keys()) if kk != '__vary_default'], dtype=object)
        value = np.array([self.line._xdeps_vref[kk]._value for kk in name])

        if compact:
            formatter = xd.refs.CompactFormatter(scope=None)
            expr = []
            for kk in name:
                ee = self.line._xdeps_vref[kk]._expr
                if ee is None:
                    expr.append(None)
                else:
                    expr.append(ee._formatted(formatter))
        else:
            expr  = [self.line._xdeps_vref[str(kk)]._expr for kk in name]
            for ii, ee in enumerate(expr):
                if ee is not None:
                    expr[ii] = str(ee)

        expr = np.array(expr)

        return VarsTable({'name': name, 'value': value, 'expr': expr})

    def new_expr(self, expr):
        return self.line._xdeps_eval.eval(expr)

    def eval(self, expr):
        expr_or_value = self.new_expr(expr)
        if is_ref(expr_or_value):
            return expr_or_value._get_value()
        return expr_or_value

    def info(self, var, limit=10):
        return self[var]._info(limit=limit)

    def get_expr(self, var):
        return self[var]._expr

    def __contains__(self, key):
        if self.line._xdeps_vref is None:
            raise RuntimeError(
                f'Cannot access variables as the line has no xdeps manager')
        return key in self.line._xdeps_vref._owner

    def get_independent_vars(self):

        """
        Returns the list of independent variables in the line.
        """

        out = []
        for kk in self.keys():
            if self[kk]._expr is None:
                out.append(kk)
        return out

    def __getitem__(self, key):
        if key not in self: # uses __contains__ method
            raise KeyError(f'Variable `{key}` not found')
        return self.line._xdeps_vref[key]

    def __setitem__(self, key, value):
        if isinstance(value, str):
            value = self.line._xdeps_eval.eval(value)
        self.line._xdeps_vref[key] = value
        for cc in self.vars_to_update:
            cc[key] = value

    def __getstate__(self):
        out = self.__dict__.copy()
        out['vars_to_update'] = None
        return out

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.vars_to_update = WeakSet()

    def set_from_madx_file(self, filename=None, string=None):

        '''
        Set variables veluas of expression from a MAD-X file.

        Parameters
        ----------
        filename : str or list of str
            Path to the MAD-X file(s) to load.
        '''
        loader = xt.mad_parser.MadxLoader(env=self.line)
        if filename is not None:
            assert string is None, 'Cannot specify both filename and string'
            loader.load_file(filename)
        elif string is not None:
            assert filename is None, 'Cannot specify both filename and string'
            loader.load_string(string)

    def load_madx_optics_file(self, filename=None, string=None):
        self.set_from_madx_file(filename, string)

    load_madx = load_madx_optics_file

    def load_json(self, filename):

        with open(filename, 'r') as fid:
            data = json.load(fid)

        _old_default_to_zero = self.default_to_zero
        self.default_to_zero = True
        self.update(data)
        self.default_to_zero = _old_default_to_zero

    def target(self, tar, value, **kwargs):
        action = ActionVars(self.line)
        return xt.Target(action=action, tar=tar, value=value, **kwargs)

    def __call__(self, *args, **kwargs):
        _eval = self.line._xdeps_eval.eval
        if len(args) > 0:
            assert len(kwargs) == 0
            assert len(args) == 1
            if isinstance(args[0], str):
                return self[args[0]]
            elif isinstance(args[0], dict):
                kwargs.update(args[0])
            else:
                raise ValueError('Invalid argument')
        for kk in kwargs:
            if isinstance(kwargs[kk], str):
                self[kk] = _eval(kwargs[kk])
            else:
                self[kk] = kwargs[kk]

    def set(self, name, value):
        if isinstance(value, str):
            self[name] = self.line._xdeps_eval.eval(value)
        else:
            self[name] = value

    def get(self, name):
        return self[name]._value

    @property
    def default_to_zero(self):
        default_factory = self.line._xdeps_vref._owner.default_factory
        if default_factory is None:
            return False
        return default_factory.default == 0

    @default_to_zero.setter
    def default_to_zero(self, value):
        assert value in (True, False)
        if value:
            self.line._xdeps_vref._owner.default_factory = _DefaultFactory(0.)
        else:
            self.line._xdeps_vref._owner.default_factory = None

class VarsTable(xd.Table):

    def to_dict(self):
        out = {}
        for nn, ee, vv in zip(self['name'], self['expr'], self['value']):
            if ee is not None:
                out[nn] = ee
            else:
                out[nn] = vv
        return out

class ActionVars(Action):

    def __init__(self, line):
        self.line = line

    def run(self, **kwargs):
        return self.line._xdeps_vref._owner

class ActionLine(Action):

    def __init__(self, line):
        self.line = line

    def run(self):
        return self.line

class VarValues:

    def __init__(self, vars):
        self.vars = vars

    def __getitem__(self, key):
        return self.vars[key]._value

    def __setitem__(self, key, value):
        self.vars[key] = value

    def get(self,key, default=0):
        if key in self.vars:
            return self.vars[key]._value
        else:
            return default

class LineAttrItem:

    def __init__(self, name, index=None, line=None):
        self.name = name
        self.index = index

        assert line is not None
        self.line = line
        self._multisetter = None

    def _prepare_multisetter(self):

        line = self.line
        name = self.name
        index = self.index

        all_names = line.element_names
        mask = np.zeros(len(all_names), dtype=bool)
        setter_names = []
        for ii, nn in enumerate(all_names):
            ee = line.element_dict[nn]
            if isinstance(ee, xt.Replica):
                nn = ee.resolve(line, get_name=True)
                ee = line.element_dict[nn]
            if isinstance(name, (list, tuple)):
                inner_obj = ee
                inner_name = name[-1]
                has_name = True
                for nn_inner in name[:-1]:
                    if not hasattr(inner_obj, nn_inner):
                        has_name = False
                        break
                    inner_obj = getattr(inner_obj, nn_inner)
                if not has_name:
                    continue
            else:
                inner_obj = ee
                inner_name = name
            if hasattr(inner_obj, '_xobject') and hasattr(inner_obj._xobject, inner_name):
                if index is not None and index >= len(getattr(inner_obj, inner_name)):
                    continue
                mask[ii] = True
                setter_names.append(nn)

        multisetter = xt.MultiSetter(line=line, elements=setter_names,
                                     field=name, index=index)
        self.names = setter_names
        self._multisetter = multisetter
        self._mask = mask

    @property
    def multisetter(self):
        if self._multisetter is None:
            self._prepare_multisetter()
        return self._multisetter

    @property
    def mask(self):
        if self._multisetter is None:
            self._prepare_multisetter()
        return self._mask

    def get_full_array(self):
        full_array = np.zeros(len(self.mask), dtype=np.float64)
        ctx2np = self.multisetter._context.nparray_from_context_array
        full_array[self.mask] = ctx2np(self.multisetter.get_values())
        return full_array

class LineAttr:
    """A class to access a field of all elements in a line.

    The field can be a scalar or a vector. In the latter case, the index
    can be specified to access a specific element of the vector.

    Parameters
    ----------
    line : Line
        The line to access.
    fields : list of str or tuple of (str, int)
        The fields to access. If a tuple is provided, the second element
        is the index of the vector to access.
    derived_fields : dict, optional
        A dictionary of derived fields. The key is the name of the derived
        field and the value is a function that takes the LineAttr object
        as argument and returns the value of the derived field.
    """

    def __init__(self, line, fields, derived_fields=None):

        assert isinstance(fields, dict)

        field_names = list(fields.keys())
        field_access = []
        for fn in field_names:
            fa = fields[fn]
            if fa is None:
                fa = fn
            field_access.append(fa)

        self.line = line
        self.fields = fields
        self.derived_fields = derived_fields or {}
        self._cache = {}

        # Build _inherit_strengths and _rot_and_shift_from_parent
        _inherit_strengths = np.zeros(len(line.element_names), dtype=np.float64)
        _rot_and_shift_from_parent = np.zeros(len(line.element_names), dtype=np.float64)
        for ii, nn in enumerate(line.element_names):
            ee = line.element_dict[nn]
            if hasattr(ee, '_inherit_strengths') and ee._inherit_strengths:
                _inherit_strengths[ii] = 1.
            if hasattr(ee, 'rot_and_shift_from_parent') and ee.rot_and_shift_from_parent:
                _rot_and_shift_from_parent[ii] = 1.
        self._inherit_strengths = _inherit_strengths
        self._rot_and_shift_from_parent = _rot_and_shift_from_parent

        for fn, fa in zip(field_names, field_access):
            if isinstance(fa, str):
                access = fa
                index = None
            else:
                access, index = fa
            self._cache[fn] = LineAttrItem(name=access, index=index, line=line)

    def __getitem__(self, key):
        if key in self.derived_fields:
            return self.derived_fields[key](self)

        return self._cache[key].get_full_array()

    def keys(self):
        return list(self.derived_fields.keys()) + list(self.fields)


class EnergyProgram:

    def __init__(self, t_s, kinetic_energy0=None, p0c=None):

        assert hasattr (t_s, '__len__'), 't_s must be a list or an array'

        assert p0c is not None or kinetic_energy0 is not None, (
            'Either p0c or kinetic_energy0 needs to be provided')

        assert np.isclose(t_s[0], 0, rtol=0, atol=1e-12), 't_s must start from 0'

        self.p0c = p0c
        self.kinetic_energy0 = kinetic_energy0
        self.t_s = t_s
        self.needs_complete = True

    def complete_init(self, line):

        assert self.needs_complete, 'EnergyProgram already completed'

        p0c = self.p0c
        kinetic_energy0 = self.kinetic_energy0
        t_s = self.t_s

        enevars = {}
        assert line is not None, 'line must be provided'
        assert line.particle_ref is not None, (
            'line must have a valid particle_ref')

        mass0 = line.particle_ref.mass0
        circumference = line.get_length()

        if p0c is not None:
            assert hasattr (p0c, '__len__'), 'p0c must be a list or an array'
            assert len(t_s) == len(p0c), 't_s and p0c must have same length'
            enevars['p0c'] = p0c

        if kinetic_energy0 is not None:
            assert hasattr (kinetic_energy0, '__len__'), (
                'kinetic_energy0 must be a list or an array')
            assert len(t_s) == len(kinetic_energy0), (
                't_s and kinetic_energy0 must have same length')

            energy0 = kinetic_energy0 + mass0
            enevars['energy0'] = energy0

        # I use a particle to make the conversions
        p = xt.Particles(**enevars, mass0=mass0)
        beta0_program = p.beta0
        bet0_mid = 0.5*(beta0_program[1:] + beta0_program[:-1])

        dt_s = np.diff(t_s)

        i_turn_at_t_samples = np.zeros_like(t_s)
        i_turn_at_t_samples[1:] = (
            beta0_program[0] * clight / circumference * t_s[0] +
            np.cumsum(bet0_mid * clight / circumference * dt_s))
        # In this way i_turn = 0 corresponds to t_s[0]

        self.t_at_turn_interpolator = xd.FunctionPieceWiseLinear(
                                x=i_turn_at_t_samples, y=t_s)
        self.p0c_interpolator = xd.FunctionPieceWiseLinear(
                                x=t_s, y=np.array(p.p0c))
        self.line = line

        self.needs_complete = False
        del self.p0c
        del self.kinetic_energy0

    def get_t_s_at_turn(self, i_turn):
        assert not self.needs_complete, 'EnergyProgram not complete'
        assert self.line is not None, 'EnergyProgram not associated to a line'
        if (i_turn > self.t_at_turn_interpolator.x[-1]).any():
            raise ValueError('`i_turn` outside program range not yet supported')
        out = self.t_at_turn_interpolator(i_turn)

        return out

    def get_p0c_at_t_s(self, t_s):
        assert not self.needs_complete, 'EnergyProgram not complete'
        assert self.line is not None, 'EnergyProgram not associated to a line'
        return self.p0c_interpolator(t_s)

    def get_beta0_at_t_s(self, t_s):
        p0c = self.get_p0c_at_t_s(t_s)
        # I use a particle to make the conversions
        p = xt.Particles(p0c=p0c, mass0=self.line.particle_ref.mass0)
        if np.isscalar(t_s):
            return p.beta0[0]
        else:
            return p.beta0

    def get_kinetic_energy0_at_t_s(self, t_s):
        p0c = self.get_p0c_at_t_s(t_s)
        # I use a particle to make the conversions
        p = xt.Particles(p0c=p0c, mass0=self.line.particle_ref.mass0)
        energy0 = p.energy0
        kinetic_energy0 = energy0 - self.line.particle_ref.mass0
        if np.isscalar(t_s):
            return kinetic_energy0[0]
        else:
            return kinetic_energy0

    def get_frev_at_t_s(self, t_s):
        beta0 = self.get_beta0_at_t_s(t_s)
        circumference = self.line.get_length()
        return beta0 * clight / circumference

    def get_p0c_increse_per_turn_at_t_s(self, t_s):

        ts_scalar = np.isscalar(t_s)
        if ts_scalar:
            t_s = np.array([t_s])

        beta0 = self.get_beta0_at_t_s(t_s)
        circumference = self.line.get_length()
        T_rev = circumference / (beta0 * clight)
        out = 0.5 * (self.get_p0c_at_t_s(t_s + T_rev)
                     - self.get_p0c_at_t_s(t_s - T_rev))

        mask_zero_neg = t_s - T_rev < 0
        if np.any(mask_zero_neg):
            out[mask_zero_neg] = (
                self.get_p0c_at_t_s(t_s[mask_zero_neg] + T_rev[mask_zero_neg])
                - self.get_p0c_at_t_s(t_s[mask_zero_neg]))

        if ts_scalar:
            out = out[0]

        return out

    @property
    def t_turn_s_line(self):
        raise ValueError('only setter allowed')

    @t_turn_s_line.setter
    def t_turn_s_line(self, value):
        p0c = self.get_p0c_at_t_s(value)
        self.line.particle_ref.update_p0c_and_energy_deviations(
                                                    p0c=p0c, update_pxpy=True)

    def to_dict(self):
        assert not self.needs_complete, 'EnergyProgram not completed'
        return {
            '__class__': self.__class__.__name__,
            't_at_turn_interpolator': self.t_at_turn_interpolator.to_dict(),
            'p0c_interpolator': self.p0c_interpolator.to_dict()}

    @classmethod
    def from_dict(cls, dct):
        self = cls.__new__(cls)
        self.t_at_turn_interpolator = xd.FunctionPieceWiseLinear.from_dict(
                                        dct['t_at_turn_interpolator'])
        self.p0c_interpolator = xd.FunctionPieceWiseLinear.from_dict(
                                        dct['p0c_interpolator'])
        self.needs_complete = False
        return self

    def copy(self, _context=None, _buffer=None, _offeset=None):
        return self.from_dict(self.to_dict())

def _vars_unused(line):
    if line._xdeps_vref is None:
        return True
    if (len(line.vars.keys()) == 2
        and '__vary_default' in line.vars.keys()
        and 't_turn_s' in line.vars.keys()):
        return True
    return False

def _angle_force_body_from_attr(attr):

    """This angle has always the curvature in the body, even for RBend elements
    with rbend_model='straight-body'. It is used mostly for plotting purposes.
    """

    weight = attr['weight']

    own_hxl = attr['_own_hxl']
    own_h = attr['_own_h']
    own_length = attr['_own_length']
    parent_hxl = attr['_parent_hxl']
    parent_h = attr['_parent_h']
    parent_length = attr['_parent_length']

    own_hxl_proper_system = own_hxl + own_h * own_length
    parent_hxl_proper_system = ((parent_hxl * weight + parent_h * parent_length * weight)
                                * attr._inherit_strengths)

    angle = own_hxl_proper_system + parent_hxl_proper_system

    return angle

def _angle_rbend_correction_from_attr(attr):

    angle = attr['_angle_force_body'].copy()

    ## Correction for RBend elements

    # Retrieve element_type from tracker cache (remove _end_point)
    element_type = attr.line.tracker._tracker_data_base._line_table.element_type[:-1]

    mask_rbend_edges = ((element_type == 'ThinSliceRBendEntry')
                        | (element_type == 'ThinSliceRBendExit'))
    mask_rbend_body_slices = ((element_type == 'ThinSliceRBend')
                            | (element_type == 'ThickSliceRBend'))
    mask_parent_is_rbend_straigth_body = (attr['_parent_rbend_model'] == 2)
    mask_rbend_edges_straight_body = (mask_rbend_edges
                                      & mask_parent_is_rbend_straigth_body)

    angle[mask_parent_is_rbend_straigth_body & mask_rbend_body_slices] = 0
    angle[mask_rbend_edges_straight_body] = 0.5 * (
        attr['_parent_h'][mask_rbend_edges_straight_body]
        * attr['_parent_length'][mask_rbend_edges_straight_body])

    return angle

def _rot_s_from_attr(attr):

    own_sin_rot_s = attr['_own_sin_rot_s'].copy()
    own_cos_rot_s = attr['_own_cos_rot_s'].copy()
    parent_sin_rot_s = attr['_parent_sin_rot_s'].copy()
    parent_cos_rot_s = attr['_parent_cos_rot_s'].copy()

    has_own_rot = (own_cos_rot_s !=0) | (own_sin_rot_s != 0)
    mask_own_rot_inactive = own_sin_rot_s < -2.
    own_cos_rot_s[mask_own_rot_inactive] = 1.
    own_sin_rot_s[mask_own_rot_inactive] = 0.

    has_parent_rot = (parent_cos_rot_s !=0) | (parent_sin_rot_s != 0)
    mask_parent_rot_inactive = parent_sin_rot_s < -2.
    parent_cos_rot_s[mask_parent_rot_inactive] = 1.
    parent_sin_rot_s[mask_parent_rot_inactive] = 0.

    rot_s_rad = 0. * own_sin_rot_s
    rot_s_rad[has_own_rot] = np.arctan2(own_sin_rot_s[has_own_rot],
                                        own_cos_rot_s[has_own_rot])
    rot_s_rad[has_parent_rot] = np.arctan2(parent_sin_rot_s[has_parent_rot],
        parent_cos_rot_s[has_parent_rot]) * attr._rot_and_shift_from_parent[has_parent_rot]

    return rot_s_rad


def _make_var_management(element_dict, dct=None):

    from collections import defaultdict

    _var_values = defaultdict(lambda: 0)
    _var_values.default_factory = None

    functions = Functions()

    manager = xd.Manager()
    _vref = manager.ref(_var_values, 'vars')
    _fref = manager.ref(functions, 'f')
    _lref = manager.ref(element_dict, 'element_refs')

    _var_management = {}
    _var_management['data'] = {}
    _var_management['data']['var_values'] = _var_values
    _var_management['data']['functions'] = functions

    _var_management['manager'] = manager
    _var_management['lref'] = _lref
    _var_management['vref'] = _vref
    _var_management['fref'] = _fref

    _vref['t_turn_s'] = 0.0

    if dct is not None:
        manager = _var_management['manager']
        for kk in dct['_var_management_data'].keys():
            data_item = dct['_var_management_data'][kk]
            if kk == 'functions':
                data_item = Functions.from_dict(data_item)
            _var_management['data'][kk].update(data_item)
        manager.load(dct['_var_manager'])

    return _var_management
