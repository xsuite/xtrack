# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import io
import math
import logging
import json
from contextlib import contextmanager
from copy import deepcopy
from pprint import pformat

import numpy as np

from . import linear_normal_form as lnf

import xobjects as xo
import xpart as xp
import xtrack as xt


from .survey import survey_from_tracker
from xtrack.twiss import (compute_one_turn_matrix_finite_differences,
                          find_closed_orbit_line, twiss_line)
from .match import match_line, closed_orbit_correction
from .tapering import compensate_radiation_energy_loss
from .mad_loader import MadLoader
from .beam_elements import element_classes
from . import beam_elements
from .beam_elements import Drift, BeamElement, Marker, Multipole
from .footprint import Footprint, _footprint_with_linear_rescale
from .internal_record import (start_internal_logging_for_elements_of_type,
                              stop_internal_logging_for_elements_of_type)

from .general import _print

log = logging.getLogger(__name__)


class Line:

    """
    Beam line object. `Line.element_names` contains the ordered list of beam
    elements, `Line.element_dict` is a dictionary associating to each name the
    corresponding beam element object.
    """

    _element_dict = None
    config = None

    def __init__(self, elements=(), element_names=None, particle_ref=None):

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
        """

        self.config = xt.tracker.TrackerConfig()
        self.config.XTRACK_MULTIPOLE_NO_SYNRAD = True
        self.config.XFIELDS_BB3D_NO_BEAMSTR = True
        self.config.XTRACK_GLOBAL_XY_LIMIT = 1.0

        # Config parameters not exposed to C code
        # (accessed by user through properties)
        self._extra_config = {}
        self._extra_config['skip_end_turn_actions'] = False
        self._extra_config['reset_s_at_end_turn'] = True
        self._extra_config['matrix_responsiveness_tol'] = lnf.DEFAULT_MATRIX_RESPONSIVENESS_TOL
        self._extra_config['matrix_stability_tol'] = lnf.DEFAULT_MATRIX_STABILITY_TOL
        self._extra_config['_radiation_model'] = None
        self._extra_config['_beamstrahlung_model'] = None
        self._extra_config['_needs_rng'] = False

        if isinstance(elements, dict):
            element_dict = elements
            if element_names is None:
                raise ValueError('`element_names` must be provided'
                                 ' if `elements` is a dictionary.')
        else:
            if element_names is None:
                element_names = [f"e{ii}" for ii in range(len(elements))]
            if len(element_names) > len(set(element_names)):
                log.warning("Repetition found in `element_names` -> renaming")
                old_element_names = element_names
                element_names = []
                counters = {nn: 0 for nn in old_element_names}
                for nn in old_element_names:
                    if counters[nn] > 0:
                        new_nn = nn + '_' + str(counters[nn])
                    else:
                        new_nn = nn
                    counters[nn] += 1
                    element_names.append(new_nn)

            assert len(element_names) == len(elements), (
                "`elements` and `element_names` should have the same length"
            )
            element_dict = dict(zip(element_names, elements))

        self.element_dict = element_dict.copy()  # avoid modifications if user provided
        self.element_names = list(element_names).copy()

        self.particle_ref = particle_ref

        self._var_management = None
        self.tracker = None

    @classmethod
    def from_dict(cls, dct, _context=None, _buffer=None, classes=()):

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

        if isinstance(dct['elements'], dict):
            elements = {}
            num_elements = len(dct['elements'].keys())
            for ii, (kk, ee) in enumerate(dct['elements'].items()):
                if ii % 100 == 0:
                    _print('Loading line from dict: '
                        f'{round(ii/num_elements*100):2d}%  ',end="\r", flush=True)
                elements[kk] = _deserialize_element(ee, class_dict, _buffer)
        elif isinstance(dct['elements'], list):
            elements = []
            num_elements = len(dct['elements'])
            for ii, ee in enumerate(dct['elements']):
                if ii % 100 == 0:
                    _print('Loading line from dict: '
                        f'{round(ii/num_elements*100):2d}%  ',end="\r", flush=True)
                elements.append(_deserialize_element(ee, class_dict, _buffer))
        else:
            raise ValueError('Field `elements` must be a dict or a list')

        self = cls(elements=elements, element_names=dct['element_names'])

        if 'particle_ref' in dct.keys():
            self.particle_ref = xp.Particles.from_dict(dct['particle_ref'],
                                    _context=_buffer.context)

        if '_var_manager' in dct.keys():
            self._init_var_management(dct=dct)

        if 'config' in dct.keys():
            self.config.data.update(dct['config'])

        if '_extra_config' in dct.keys():
            self._extra_config.update(dct['_extra_config'])

        _print('Done loading line from dict.           ')

        return self

    @classmethod
    def from_json(cls, file, **kwargs):

        """Constructs a line from a json file.

        Parameters
        ----------
        file : str or file-like object
            Path to the json file or file-like object.
        **kwargs : dict
            Additional keyword arguments passed to `Line.from_dict`.

        Returns
        -------
        line : Line
            Line object.

        """

        if isinstance(file, io.IOBase):
            dct = json.load(file)
        else:
            with open(file, 'r') as fid:
                dct = json.load(fid)

        if 'line' in dct.keys():
            dct_line = dct['line']
        else:
            dct_line = dct

        return cls.from_dict(dct_line, **kwargs)

    @classmethod
    def from_sequence(cls, nodes=None, length=None, elements=None,
                      sequences=None, copy_elements=False,
                      naming_scheme='{}{}', auto_reorder=False, **kwargs):

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
            defined in any order and are re-ordered as neseccary. Useful to
            place additional elements inside of sub-sequences.
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
            if node.s < last_s:
                raise ValueError(f'Negative drift space from {last_s} to {node.s}'
                    f' ({node.name}). Fix or set auto_reorder=True')
            if _is_thick(node.what):
                raise NotImplementedError(
                    f'Thick elements currently not implemented: {node.name}')

            # insert drift as needed (re-use if possible)
            if node.s > last_s:
                ds = node.s - last_s
                if ds not in drifts:
                    drifts[ds] = Drift(length=ds)
                element_objects.append(drifts[ds])
                element_names.append(_next_name('drift', element_names, naming_scheme))

            # insert element
            element_objects.append(node.what)
            element_names.append(node.name)
            last_s = node.s

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
        apply_madx_errors=False,
        skip_markers=False,
        merge_drifts=False,
        merge_multipoles=False,
        expressions_for_element_types=None,
        replace_in_expr=None,
        classes=(),
        ignored_madtypes=(),
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

        Returns
        -------
        line : Line
            Line object.

        """

        class_namespace=mk_class_namespace(classes)

        loader = MadLoader(sequence,
            classes=class_namespace,
            ignore_madtypes=ignored_madtypes,
            enable_errors=apply_madx_errors,
            enable_apertures=install_apertures,
            enable_expressions=deferred_expressions,
            skip_markers=skip_markers,
            merge_drifts=merge_drifts,
            merge_multipoles=merge_multipoles,
            expressions_for_element_types=expressions_for_element_types,
            error_table=None,  # not implemented yet
            replace_in_expr=replace_in_expr
            )
        line=loader.make_line()
        return line

    def to_dict(self, include_var_management=True):

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
        out["elements"] = {k: el.to_dict() for k, el in self.element_dict.items()}
        out["element_names"] = self.element_names[:]
        out['config'] = self.config.data.copy()
        out['_extra_config'] = self._extra_config.copy()
        if self.particle_ref is not None:
            out['particle_ref'] = self.particle_ref.to_dict()
        if self._var_management is not None and include_var_management:
            if hasattr(self, '_in_multiline') and self._in_multiline:
                raise ValueError('The line is part ot a MultiLine object. '
                    'To save without expressions please use '
                    '`line.to_dict(include_var_management=False)`.\n'
                    'To save also the deferred expressions please save the '
                    'entire multiline.\n ')

            out.update(self._var_management_to_dict())
        return out

    def to_json(self, file, **kwargs):
        '''Save the line to a json file.

        Parameters
        ----------
        file: str or file-like object
            The file to save to. If a string is provided, a file is opened and
            closed. If a file-like object is provided, it is used directly.
        **kwargs: dict
            Additional keyword arguments are passed to the `Line.to_dict` method.

        '''

        if isinstance(file, io.IOBase):
            json.dump(self.to_dict(**kwargs), file, cls=xo.JEncoder)
        else:
            with open(file, 'w') as fid:
                json.dump(self.to_dict(**kwargs), fid, cls=xo.JEncoder)

    def to_pandas(self):
        '''
        Return a pandas DataFrame with the elements of the line.

        Returns
        -------
        line_df : pandas.DataFrame
            DataFrame with the elements of the line.
        '''

        elements = self.elements
        s_elements = np.array(self.get_s_elements())
        element_types = list(map(lambda e: e.__class__.__name__, elements))
        isthick = np.array(list(map(_is_thick, elements)))

        import pandas as pd

        elements_df = pd.DataFrame({
            'element_type': element_types,
            's': s_elements,
            'name': self.element_names,
            'isthick': isthick,
            'element': elements
        })
        return elements_df

    def copy(self, _context=None, _buffer=None):
        '''
        Return a copy of the line.

        Parameters
        ----------
        _context: xobjects.Context
            xobjects context to be used for the copy
        _buffer: xobjects.Buffer
            xobjects buffer to be used for the copy

        Returns
        -------
        line_copy : Line
            Copy of the line.
        '''

        elements = {nn: ee.copy(_context=_context, _buffer=_buffer)
                                    for nn, ee in self.element_dict.items()}
        element_names = [nn for nn in self.element_names]

        out = self.__class__(elements=elements, element_names=element_names)

        if self.particle_ref is not None:
            out.particle_ref = self.particle_ref.copy(
                                        _context=_context, _buffer=_buffer)

        if self._var_management is not None:
            out._init_var_management(dct=self._var_management_to_dict())

        out.config.update(self.config.copy())
        out._extra_config.update(self._extra_config.copy())

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

        assert self.tracker is None, 'The line already has an associated tracker'
        import xtrack as xt  # avoid circular import
        self.tracker = xt.Tracker(
                                line=self,
                                _context=_context,
                                _buffer=_buffer,
                                compile=compile,
                                io_buffer=io_buffer,
                                use_prebuilt_kernels=use_prebuilt_kernels,
                                enable_pipeline_hold=enable_pipeline_hold,
                                **kwargs)

        return self.tracker

    def discard_tracker(self):

        """
        Discard the tracker associated to the line. This unfreezes the line
        (elements can be inserted or removed again).

        """

        self.element_names = list(self.element_names)
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

        """

        self._check_valid_tracker()
        return self.tracker._track(
            particles,
            ele_start=ele_start,
            ele_stop=ele_stop,
            num_elements=num_elements,
            num_turns=num_turns,
            turn_by_turn_monitor=turn_by_turn_monitor,
            freeze_longitudinal=freeze_longitudinal,
            time=time,
            **kwargs)

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
        particles_class=None,
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
            x coordinate of the particles (default is 0).
        px : float or array
            px coordinate of the particles (default is 0).
        y : float or array
            y coordinate of the particles (default is 0).
        py : float or array
            py coordinate of the particles (default is 0).
        zeta : float or array
            zeta coordinate of the particles (default is 0).
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
            `s` location within the line at which particles are generated. The value
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

        return xp.build_particles(
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
            particles_class=particles_class,
            _context=_context, _buffer=_buffer, _offset=_offset,
            _capacity=_capacity,
            mode=mode,
            **kwargs)

    def twiss(self, particle_ref=None, delta0=None, zeta0=None, method='6d',
        r_sigma=0.01, nemitt_x=1e-6, nemitt_y=1e-6,
        delta_disp=1e-5, delta_chrom=1e-4,
        particle_co_guess=None, R_matrix=None, W_matrix=None,
        steps_r_matrix=None, co_search_settings=None, at_elements=None, at_s=None,
        values_at_element_exit=False,
        continue_on_closed_orbit_error=False,
        freeze_longitudinal=False,
        radiation_method='full',
        eneloss_and_damping=False,
        ele_start=None, ele_stop=None, twiss_init=None,
        particle_on_co=None,
        matrix_responsiveness_tol=None,
        matrix_stability_tol=None,
        symplectify=False,
        reverse=False,
        use_full_inverse=None,
        strengths=False,
        hide_thin_groups=False,
        ):

        self._check_valid_tracker()

        kwargs = locals().copy()
        kwargs.pop('self')

        return twiss_line(self, **kwargs)
    twiss.__doc__ = twiss_line.__doc__

    def match(self, vary, targets, restore_if_fail=True, solver=None,
                  verbose=False, **kwargs):
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
        restore_if_fail : bool
            If True, the beamline is restored to its initial state if the matching
            fails.
        solver : str
            Solver to be used for the matching. Available solvers are "fsolve"
            and "bfgs".
        verbose : bool
            If True, the matching steps are printed.
        **kwargs : dict
            Additional arguments to be passed to the twiss.

        Returns
        -------
        result_info : dict
            Dictionary containing information about the matching result.

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
                ele_start='mq.33l8.b1',
                ele_stop='mq.23l8.b1',
                twiss_init=tw_before.get_twiss_init(at_element='mq.33l8.b1'),
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
        return match_line(self, vary, targets,
                          restore_if_fail=restore_if_fail,
                          solver=solver, verbose=verbose, **kwargs)

    def survey(self,X0=0,Y0=0,Z0=0,theta0=0, phi0=0, psi0=0,
               element0=0, reverse=False):

        """
        Returns a survey of the beamline (based on MAD-X survey command).

        Parameters
        ----------
        X0 : float
            Initial X coordinate.
        Y0 : float
            Initial Y coordinate.
        Z0 : float
            Initial Z coordinate.
        theta0 : float
            Initial theta coordinate.
        phi0 : float
            Initial phi coordinate.
        psi0 : float
            Initial psi coordinate.
        element0 : int or str
            Element at which the given coordinates are defined.

        Returns
        -------
        survey : SurveyTable
            Survey table.
        """

        return survey_from_tracker(self.tracker, X0=X0, Y0=Y0, Z0=Z0, theta0=theta0,
                                   phi0=phi0, psi0=psi0, element0=element0,
                                   reverse=reverse)

    def correct_closed_orbit(self, reference, correction_config,
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

    def find_closed_orbit(self, particle_co_guess=None, particle_ref=None,
                          co_search_settings={}, delta_zeta=0,
                          delta0=None, zeta0=None,
                          continue_on_closed_orbit_error=False,
                          freeze_longitudinal=False):

        """
        Find the closed orbit of the beamline.

        Parameters
        ----------
        particle_co_guess : Particle
            Particle used to compute the closed orbit. If None, the reference
            particle is used.
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
            Initial zeta coordinate.
        continue_on_closed_orbit_error : bool
            If True, the closed orbit at the last step is returned even if
            the closed orbit search fails.
        freeze_longitudinal : bool
            If True, the longitudinal coordinates are frozen during the closed
            orbit search.

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

        if particle_ref is None and particle_co_guess is None:
            particle_ref = self.particle_ref

        if self.iscollective:
            log.warning(
                'The tracker has collective elements.\n'
                'In the twiss computation collective elements are'
                ' replaced by drifts')
            line = self._get_non_collective_line()
        else:
            line = self

        return find_closed_orbit_line(line, particle_co_guess=particle_co_guess,
                                 particle_ref=particle_ref, delta0=delta0, zeta0=zeta0,
                                 co_search_settings=co_search_settings, delta_zeta=delta_zeta,
                                 continue_on_closed_orbit_error=continue_on_closed_orbit_error)

    def get_footprint(self, nemitt_x=None, nemitt_y=None, n_turns=256, n_fft=2**18,
            mode='polar', r_range=None, theta_range=None, n_r=None, n_theta=None,
            x_norm_range=None, y_norm_range=None, n_x_norm=None, n_y_norm=None,
            linear_rescale_on_knobs=None,
            freeze_longitudinal=None, delta0=None, zeta0=None,
            keep_fft=True):

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
            Range of theta values for footprint in polar mode. Default is
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
            Initial value of the zeta coordinate.

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

    def compute_one_turn_matrix_finite_differences(
            self, particle_on_co,
            steps_r_matrix=None):

        '''Compute the one turn matrix using finite differences.

        Parameters
        ----------
        particle_on_co : Particle
            Particle at the closed orbit.
        steps_r_matrix : float
            Step size for finite differences. In not given, default step sizes
            are used.

        Returns
        -------
        one_turn_matrix : np.ndarray
            One turn matrix.

        '''

        self._check_valid_tracker()

        if self.iscollective:
            log.warning(
                'The tracker has collective elements.\n'
                'In the twiss computation collective elements are'
                ' replaced by drifts')
            line = self._get_non_collective_line()
        else:
            line = self

        return compute_one_turn_matrix_finite_differences(line, particle_on_co,
                                                          steps_r_matrix)


    def get_length(self):

        '''Get total length of the line'''

        ll = 0
        for ee in self.elements:
            if _is_thick(ee):
                ll += ee.length

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
        s_prev = 0
        s = []
        for ee in self.elements:
            if mode == "upstream":
                s.append(s_prev)
            if _is_thick(ee):
                s_prev += ee.length
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

    def insert_element(self, index=None, element=None, name=None, at_s=None,
                       s_tol=1e-6):

        '''Insert an element in the line.

        Parameters
        ----------
        index: int, optional
            Index of the element in the line. If `index` is provided, `at_s`
            must be None.
        element: xline.Element, optional
            Element to be inserted. If `element` is provided, `name` must be
            provided.
        name: str
            Name of the element. If `name` is provided, `element` must be
            provided.
        at_s: float, optional
            Position of the element in the line. If `at_s` is provided, `index`
            must be None.
        s_tol: float, optional
            Tolerance for the position of the element in the line.
        '''

        if isinstance(index, str):
            assert index in self.element_names
            index = self.element_names.index(index)

        assert name is not None
        if element is None:
            assert name in self.element_names
            element  = self.element_dict[name]

        self._frozen_check()

        assert ((index is not None and at_s is None) or
                (index is None and at_s is not None)), (
                    "Either `index` or `at_s` must be provided"
                )

        if at_s is not None:
            s_vect_upstream = np.array(self.get_s_position(mode='upstream'))

            if not _is_thick(element) or np.abs(element.length)==0:
                i_closest = np.argmin(np.abs(s_vect_upstream - at_s))
                if np.abs(s_vect_upstream[i_closest] - at_s) < s_tol:
                    return self.insert_element(index=i_closest,
                                            element=element, name=name)

            s_vect_downstream = np.array(self.get_s_position(mode='downstream'))

            s_start_ele = at_s
            i_first_drift_to_cut = np.where(s_vect_downstream > s_start_ele)[0][0]

            # Shortcut for thin element without drift splitting
            if (not _is_thick(element)
                and np.abs(s_vect_upstream[i_first_drift_to_cut]-at_s)<1e-10):
                    return self.insert_element(index=i_first_drift_to_cut,
                                              element=element, name=name)

            if _is_thick(element) and np.abs(element.length)>0:
                s_end_ele = at_s + element.length
            else:
                s_end_ele = s_start_ele

            i_last_drift_to_cut = np.where(s_vect_upstream < s_end_ele)[0][-1]
            if _is_thick(element) and element.length > 0:
                assert i_first_drift_to_cut <= i_last_drift_to_cut
            name_first_drift_to_cut = self.element_names[i_first_drift_to_cut]
            name_last_drift_to_cut = self.element_names[i_last_drift_to_cut]
            first_drift_to_cut = self.element_dict[name_first_drift_to_cut]
            last_drift_to_cut = self.element_dict[name_last_drift_to_cut]

            assert _is_drift(first_drift_to_cut)
            assert _is_drift(last_drift_to_cut)

            for ii in range(i_first_drift_to_cut, i_last_drift_to_cut+1):
                e_to_replace = self.element_dict[self.element_names[ii]]
                if (not _is_drift(e_to_replace) and
                    not isinstance(e_to_replace, Marker) and
                    not _is_aperture(e_to_replace)):
                    raise ValueError('Cannot replace active element '
                                        f'{self.element_names[ii]}')

            l_left_part = s_start_ele - s_vect_upstream[i_first_drift_to_cut]
            l_right_part = s_vect_downstream[i_last_drift_to_cut] - s_end_ele
            assert l_left_part >= 0
            assert l_right_part >= 0
            name_left = name_first_drift_to_cut + '_part0'
            name_right = name_last_drift_to_cut + '_part1'

            self.element_names[i_first_drift_to_cut:i_last_drift_to_cut] = []
            i_insert = i_first_drift_to_cut

            drift_base = self.element_dict[self.element_names[i_insert]]
            drift_left = drift_base.copy()
            drift_left.length = l_left_part
            drift_right = drift_base.copy()
            drift_right.length = l_right_part

            # Insert
            assert name_left not in self.element_names
            assert name_right not in self.element_names

            names_to_insert = []

            if drift_left.length > 0:
                names_to_insert.append(name_left)
                self.element_dict[name_left] = drift_left
            names_to_insert.append(name)
            self.element_dict[name] = element
            if drift_right.length > 0:
                names_to_insert.append(name_right)
                self.element_dict[name_right] = drift_right

            self.element_names[i_insert] = names_to_insert[-1]
            if len(names_to_insert) > 1:
                for nn in names_to_insert[:-1][::-1]:
                    self.element_names.insert(i_insert, nn)

        else:
            if _is_thick(element) and np.abs(element.length)>0:
                raise NotImplementedError('use `at_s` to insert thick elements')
            assert name not in self.element_dict.keys()
            self.element_dict[name] = element
            self.element_names.insert(index, name)

        return self

    def append_element(self, element, name):

        '''Append element to the end of the lattice

        Parameters
        ----------
        element : object
            Element to append
        name : str
            Name of the element to append
        '''

        self._frozen_check()
        assert name not in self.element_dict.keys()
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

        new_elements = []
        assert len(mask) == len(self.elements)
        for ff, ee in zip(mask, self.elements):
            if ff:
                new_elements.append(ee)
            else:
                if _is_thick(ee) and not _is_drift(ee):
                    new_elements.append(Drift(length=ee.length))
                else:
                    new_elements.append(Drift(length=0))

        new_line = self.__class__(elements=new_elements,
                              element_names=self.element_names)
        if self.particle_ref is not None:
            new_line.particle_ref = self.particle_ref.copy()

        if self._has_valid_tracker():
            new_line.build_tracker(_buffer=self._buffer,
                                   track_kernel=self.tracker.track_kernel,
                                   element_classes=self.tracker.element_classes)
            #TODO: handle config and other metadata

        return new_line

    def cycle(self, index_first_element=None, name_first_element=None,
              inplace=False):

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

        if ((index_first_element is not None and name_first_element is not None)
               or (index_first_element is None and name_first_element is None)):
             raise ValueError(
                "Please provide either `index_first_element` or `name_first_element`.")

        if type(index_first_element) is str:
            name_first_element = index_first_element
            index_first_element = None

        if name_first_element is not None:
            assert self.element_names.count(name_first_element) == 1, (
                f"name_first_element={name_first_element} occurs more than once!"
            )
            index_first_element = self.element_names.index(name_first_element)

        new_element_names = (list(self.element_names[index_first_element:])
                             + list(self.element_names[:index_first_element]))

        has_valid_tracker = self._has_valid_tracker()
        if has_valid_tracker:
            buffer = self._buffer
            track_kernel = self.tracker.track_kernel
            element_classes = self.tracker.element_classes
        else:
            buffer = None
            track_kernel = None
            element_classes = None

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
                                   track_kernel=track_kernel,
                                   element_classes=element_classes)
            #TODO: handle config and other metadata

        return new_line

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
            self.freeze_vars(xp.Particles.part_energy_varnames() + ['zeta'])
        else:
            self.unfreeze_vars(xp.Particles.part_energy_varnames() + ['zeta'])

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

    def configure_radiation(self, model=None, model_beamstrahlung=None,
                            mode='deprecated'):

        """
        Configure radiation within the line.

        Parameters
        ----------
        model: str
            Radiation model to use. Can be 'mean', 'quantum' or None.
        model_beamstrahlung: str
            Beamstrahlung model to use. Can be 'mean', 'quantum' or None.

        """

        if mode != 'deprecated':
            raise NameError('mode is deprecated, use model instead')

        self._check_valid_tracker()

        assert model in [None, 'mean', 'quantum']
        assert model_beamstrahlung in [None, 'mean', 'quantum']

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

        for kk, ee in self.element_dict.items():
            if hasattr(ee, 'radiation_flag'):
                ee.radiation_flag = radiation_flag

        for kk, ee in self.element_dict.items():
            if hasattr(ee, 'flag_beamstrahlung'):
                ee.flag_beamstrahlung = beamstrahlung_flag

        if radiation_flag == 2 or beamstrahlung_flag == 2:
            self._needs_rng = True

        self.config.XTRACK_MULTIPOLE_NO_SYNRAD = (radiation_flag == 0)
        self.config.XFIELDS_BB3D_NO_BEAMSTR = (beamstrahlung_flag == 0)

    def compensate_radiation_energy_loss(self, delta0=0, rtol_eneloss=1e-10,
                                    max_iter=100, **kwargs):

        """
        Compensate beam energy loss from synchrotron radiation by configuring
        RF cavities and Multipole elements (tapering).

        Parameters
        ----------
        delta0: float
            Initial energy deviation.
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

        # Unfreeze the line
        self.element_names = list(self.element_names)

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
        tracker_data = xt.tracker_data.TrackerData(
            element_dict=self.element_dict,
            element_names=self.element_names,
            element_s_locations=self.get_s_elements(),
            line_length=self.get_length(),
            extra_element_classes=(self.tracker.particles_monitor_class._XoStruct,),
            _buffer=self._buffer)

        self._freeze()

        self.tracker._tracker_data = tracker_data
        self.tracker._element_classes = tracker_data.element_classes

        self.use_prebuilt_kernels = False

        if compile:
            _ = self.tracker._current_track_kernel # This triggers compilation

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

    def remove_markers(self, inplace=True, keep=None):
        '''
        Remove markers from the line

        Parameters
        ----------
        inplace : bool
            If True, remove markers from the line (default: True)
        keep : str or list of str
            Name of the markers to keep (default: None)
        '''

        if self._var_management is not None:
            raise NotImplementedError('`remove_markers` not'
                                      ' available when deferred expressions are'
                                      ' used')

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
            self.element_dict = newline.element_dict
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

        if self._var_management is not None:
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
                aux = ([ee.hxl, ee.hyl]
                        + list(ctx2np(ee.knl)) + list(ctx2np(ee.ksl)))
                if np.sum(np.abs(np.array(aux))) == 0.0:
                    continue
            newline.append_element(ee, nn)

        if inplace:
            self.element_names = newline.element_names
            self.element_dict = newline.element_dict
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

        if self._var_management is not None:
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
            if _is_drift(ee) and nn not in keep:
                if ee.length == 0.0:
                    continue
            newline.append_element(ee, nn)

        if inplace:
            self.element_names = newline.element_names
            self.element_dict = newline.element_dict
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

        if self._var_management is not None:
            raise NotImplementedError('`merge_consecutive_drifts` not'
                                      ' available when deferred expressions are'
                                      ' used')

        self._frozen_check()

        if keep is None:
            keep = []
        elif isinstance(keep, str):
            keep = [keep]

        newline = Line(elements=[], element_names=[])

        for ii, (ee, nn) in enumerate(zip(self.elements, self.element_names)):
            if ii == 0:
                newline.append_element(ee.copy(), nn)
                continue

            this_ee = ee if inplace else ee.copy()
            if _is_drift(ee) and not nn in keep:
                prev_nn = newline.element_names[-1]
                prev_ee = newline.element_dict[prev_nn]
                if _is_drift(prev_ee) and not prev_nn in keep:
                    prev_ee.length += ee.length
                else:
                    newline.append_element(this_ee, nn)
            else:
                newline.append_element(this_ee, nn)

        if inplace:
            self.element_names = newline.element_names
            self.element_dict = newline.element_dict
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

        # For every occurence of three or more apertures that are the same,
        # only separated by Drifts or Markers, this script removes the
        # middle apertures
        # TODO: this probably actually works, but better be safe than sorry
        if self._var_management is not None:
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
            if ee.__class__.__name__.startswith('Limit'):
            # We encountered a new aperture, shift all previous
                aper_m2 = aper_m1
                aper_m1 = aper_0
                aper_0  = nn
            elif (not isinstance(ee, (Drift, Marker))
            or nn in drifts_that_need_aperture):
            # We are in an active element: all previous apertures
            # should be kept in the line
                aper_0  = None
                aper_m1 = None
                aper_m2 = None
            if (aper_m2 is not None
                and _apertures_equal(self.element_dict[aper_0], self.element_dict[aper_m1])
                and _apertures_equal(self.element_dict[aper_m1], self.element_dict[aper_m2])
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
                    knl=element.knl,
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
                    knl=element.knl,
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
                                            lambda nn: _is_aperture(self.element_dict[nn]))
        elements_df['i_aperture_upstream'] = np.nan
        elements_df['s_aperture_upstream'] = np.nan
        elements_df['i_aperture_downstream'] = np.nan
        elements_df['s_aperture_downstream'] = np.nan
        num_elements = len(self.element_names)

        # Elements that don't need aperture
        dont_need_aperture = {name: False for name in elements_df['name']}
        for name in elements_df['name']:
            ee = self.element_dict[name]
            if _allow_backtrack(ee) and not name in needs_aperture:
                dont_need_aperture[name] = True

            # Correct isthick for elements that need aperture but have zero length.
            # Use-case example: Before collimators are installed as EverestCollimator
            # (or any BaseCollimator element), they are just Markers or Drifts. We
            # want to enforce that they have an aperture when loading the line (when
            # they are still Drifts), so their names are added to 'needs_aperture'.
            # However, it is enough for them to have an upstream aperture as they are
            # at this stage just Markers (and xcoll takes care of providing the down-
            # stream aperture), so we mark them as thin.
            if name in needs_aperture and hasattr(ee, 'length') and ee.length == 0:
                elements_df.loc[elements_df['name']==name, 'isthick'] = False

        i_prev_aperture = elements_df[elements_df['is_aperture']].index[0]
        i_next_aperture = 0

        for iee in range(i_prev_aperture, num_elements):

            if iee % 100 == 0:
                _print(
                    f'Checking aperture: {round(iee/num_elements*100):2d}%  ',
                    end="\r", flush=True)

            if dont_need_aperture[elements_df.loc[iee, 'name']]:
                continue

            if elements_df.loc[iee, 'is_aperture']:
                i_prev_aperture = iee
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
        s_downstream.loc[df_thick_to_check.index] += np.array([ee.length for ee in df_thick_to_check.element])
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

        if self._var_management is not None:
            raise NotImplementedError('`merge_consecutive_multipoles` not'
                                      ' available when deferred expressions are'
                                      ' used')

        self._frozen_check()

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
                    and prev_ee.hxl==ee.hxl==0 and prev_ee.hyl==ee.hyl==0
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
                            knl=knl, ksl=ksl, hxl=prev_ee.hxl, hyl=prev_ee.hyl,
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
            self.element_dict = newline.element_dict
            return self
        else:
            return newline

    def get_backtracker(self, _context=None, _buffer=None):

        """
        Get a backtracker for this line.

        Parameters
        ----------
        _context : xobjects.Context, optional
            The context to use for the backtracker. If None, the default
            context is used.
        _buffer : xobjects.Buffer, optional
            The buffer to use for the backtracker. If None, a new buffer is
            created from the context.

        Returns
        -------
        line : Line
            The modified line.

        """

        self._check_valid_tracker()
        backtracker = self.tracker.get_backtracker(_context=_context,
                                                   _buffer=_buffer)
        return backtracker.line

    def _freeze(self):
        self.element_names = tuple(self.element_names)

    def unfreeze(self):

        # Unfreeze the line. This is useful if you want to modify the line
        # after it has been frozen (most likely by calling `build_tracker`).

        self.discard_tracker()

    def _frozen_check(self):
        if isinstance(self.element_names, tuple):
            raise ValueError(
                'This action is not allowed as the line is frozen!')

    def __len__(self):
        return len(self.element_names)

    def _var_management_to_dict(self):
        out = {}
        out['_var_management_data'] = deepcopy(self._var_management['data'])
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
                "This tracker is not anymore valid, most probably because the corresponding line has been unfrozen. "
                "Please rebuild the tracker, for example using `line.build_tracker(...)`.")

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

        from collections import defaultdict
        import xdeps as xd

        _var_values = defaultdict(lambda: 0)
        _var_values.default_factory = None

        manager = xd.Manager()
        _vref = manager.ref(_var_values, 'vars')
        _fref = manager.ref(mathfunctions, 'f')
        _lref = manager.ref(self.element_dict, 'element_refs')

        self._var_management = {}
        self._var_management['data'] = {}
        self._var_management['data']['var_values'] = _var_values

        self._var_management['manager'] = manager
        self._var_management['lref'] = _lref
        self._var_management['vref'] = _vref
        self._var_management['fref'] = _fref

        if dct is not None:
            manager = self._var_management['manager']
            for kk in self._var_management['data'].keys():
                self._var_management['data'][kk].update(
                                            dct['_var_management_data'][kk])
            manager.load(dct['_var_manager'])

    @property
    def record_last_track(self):
        self._check_valid_tracker()
        return self.tracker.record_last_track

    @property
    def vars(self):
        if self._var_management is not None:
            return self._var_management['vref']

    @property
    def element_refs(self):
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
    def _beamstrahlung_model(self):
        return self._extra_config['_beamstrahlung_model']

    @_beamstrahlung_model.setter
    def _beamstrahlung_model(self, value):
        self._extra_config['_beamstrahlung_model'] = value

    @property
    def _needs_rng(self):
        return self._extra_config['_needs_rng']

    @_needs_rng.setter
    def _needs_rng(self, value):
        self._extra_config['_needs_rng'] = value

    @property
    def time_last_track(self):
        self._check_valid_tracker()
        return self.tracker.time_last_track

    def __getitem__(self, ii):
        if isinstance(ii, str):
            if ii not in self.element_names:
                raise IndexError(f'No installed element with name {ii}')
            return self.element_dict.__getitem__(ii)
        else:
            names = self.element_names.__getitem__(ii)
            if isinstance(names, str):
                return self.element_dict.__getitem__(names)
            else:
                return [self.element_dict[nn] for nn in names]

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

mathfunctions = type('math', (), {})
mathfunctions.sqrt = math.sqrt
mathfunctions.log = math.log
mathfunctions.log10 = math.log10
mathfunctions.exp = math.exp
mathfunctions.sin = math.sin
mathfunctions.cos = math.cos
mathfunctions.tan = math.tan
mathfunctions.asin = math.asin
mathfunctions.acos = math.acos
mathfunctions.atan = math.atan
mathfunctions.sinh = math.sinh
mathfunctions.cosh = math.cosh
mathfunctions.tanh = math.tanh
mathfunctions.sinc = np.sinc
mathfunctions.abs = math.fabs
mathfunctions.erf = math.erf
mathfunctions.erfc = math.erfc
mathfunctions.floor = math.floor
mathfunctions.ceil = math.ceil
mathfunctions.round = np.round
mathfunctions.frac = lambda x: (x % 1)


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
    return (el.radiation_flag == 0 and
            el.order == 1 and
            el.knl[0] == 0 and
            el.length == 0 and
            not any(el.ksl) and
            not el.hxl and
            not el.hyl)


def _is_simple_dipole(el):
    if not isinstance(el, Multipole):
        return False
    return (el.radiation_flag == 0 and el.order == 0
            and not any(el.ksl) and not el.hyl)


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
        log.warning("Xfields not installed correctly")

    out = AttrDict()
    for cl in all_classes:
        out[cl.__name__] = cl
    return out


def _is_drift(element): # can be removed if length is zero
    return isinstance(element, (beam_elements.Drift,) )

def _behaves_like_drift(element):
    return hasattr(element, 'behaves_like_drift') and element.behaves_like_drift

def _is_aperture(element):
    return element.__class__.__name__.startswith('Limit')

def _is_thick(element):
    return  hasattr(element, "isthick") and element.isthick

def _allow_backtrack(element):
    return hasattr(element, 'allow_backtrack') and element.allow_backtrack

def _next_name(prefix, names, name_format='{}{}'):
    """Return an available element name by appending a number"""
    if prefix not in names: return prefix
    i = 1
    while name_format.format(prefix, i) in names:
        i += 1
    return name_format.format(prefix, i)

def _dicts_equal(dict1, dict2):
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        raise ValueError
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

def _apertures_equal(ap1, ap2):
    if not _is_aperture(ap1) or not _is_aperture(ap2):
        raise ValueError(f"Element {ap1} or {ap2} not an aperture!")
    if ap1.__class__ != ap2.__class__:
        return False
    ap1 = ap1.to_dict()
    ap2 = ap2.to_dict()
    return _dicts_equal(ap1, ap2)

def _lines_equal(line1, line2):
    return _dicts_equal(line1.to_dict(), line2.to_dict())


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