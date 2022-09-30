# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import math
import logging
from copy import deepcopy
from pprint import pp

import numpy as np

import xobjects as xo
import xpart as xp

from .mad_loader import MadLoader
from .beam_elements import element_classes
from . import beam_elements
from .beam_elements import Drift

log=logging.getLogger(__name__)

def mk_class_namespace(extra_classes):
    try:
        import xfields as xf
        all_classes = element_classes + xf.element_classes + extra_classes + (Line,)
    except ImportError:
        all_classes = element_classes + extra_classes
        log.warning("Xfields not installed correctly")

    out=AttrDict()
    for cl in all_classes:
        out[cl.__name__]=cl
    return out


_thick_element_types = (beam_elements.Drift, ) #TODO add DriftExact

def _is_drift(element): # can be removed if length is zero
    return isinstance(element, (beam_elements.Drift,) )

def _is_thick(element):
    return  ((hasattr(element, "isthick") and element.isthick) or
             (isinstance(element, _thick_element_types)))

DEG2RAD = np.pi / 180.

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Line:

    @classmethod
    def from_dict(cls, dct, _context=None, _buffer=None, classes=()):
        class_dict = mk_class_namespace(classes)

        _buffer = xo.get_a_buffer(context=_context, buffer=_buffer,size=8)

        if isinstance(dct['elements'], dict):
            elements = {}
            num_elements = len(dct['elements'].keys())
            for ii, (kk, ee) in enumerate(dct['elements'].items()):
                if ii % 100 == 0:
                    print('Loading line from dict: '
                        f'{round(ii/num_elements*100):2d}%  ',end="\r", flush=True)
                elements[kk] = _deserialize_element(ee, class_dict, _buffer)
        elif isinstance(dct['elements'], list):
            elements = []
            num_elements = len(dct['elements'])
            for ii, ee in enumerate(dct['elements']):
                if ii % 100 == 0:
                    print('Loading line from dict: '
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

        print('Done loading line from dict.           ')

        return self

    @classmethod
    def from_sixinput(cls, sixinput, classes=()):
        log.warning("\n"
            "WARNING: xtrack.Line.from_sixinput(sixinput) will be removed in future versions.\n"
            "Please use sixinput.generate_xtrack_line()\n")
        line = sixinput.generate_xtrack_line(classes=classes)
        return line

    @classmethod
    def from_madx_sequence(
        cls,
        sequence,
        classes=(),
        ignored_madtypes=(),
        exact_drift=False,
#        drift_threshold=1e-6, # not used anymore with expanded sequences
        deferred_expressions=False,
        install_apertures=False,
        apply_madx_errors=False,
        skip_markers=False,
        merge_drifts=False,
        merge_multipoles=False,
    ):

        if not (exact_drift is False):
            raise NotImplementedError("Exact drifts not implemented yet")

        class_namespace=mk_class_namespace(classes)

        loader = MadLoader(sequence,
            classes=class_namespace,
            ignore_madtypes=ignored_madtypes,
            exact_drift=False,
            enable_errors=apply_madx_errors,
            enable_apertures=install_apertures,
            enable_expressions=deferred_expressions,
            skip_markers=skip_markers,
            merge_drifts=merge_drifts,
            merge_multipoles=merge_multipoles,
            error_table=None,  # not implemented yet
            )
        line=loader.make_line()
        return line

    def _init_var_management(self, dct=None):

        from collections import defaultdict
        import xdeps as xd

        # Extract globals values from madx
        _var_values = defaultdict(lambda :0)
        _var_values.default_factory = None

        _ref_manager = manager=xd.Manager()
        _vref=manager.ref(_var_values,'vars')
        _fref=manager.ref(mathfunctions,'f')
        _lref = manager.ref(self.element_dict, 'element_refs')

        self._var_management = {}
        self._var_management['data'] = {}
        self._var_management['data']['var_values'] = _var_values

        self._var_management['manager'] = _ref_manager
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
    def vars(self):
        if self._var_management is not None:
            return self._var_management['vref']

    @property
    def element_refs(self):
        if self._var_management is not None:
            return self._var_management['lref']

    def __init__(self, elements=(), element_names=None, particle_ref=None):
        if isinstance(elements,dict):
            element_dict=elements
            if element_names is None:
                raise ValueError('`element_names must be provided'
                                 ' if `elements` is a dictionary.')
        else:
            if element_names is None:
                element_names = [ f"e{ii}" for ii in range(len(elements))]
            if len(element_names) > len(set(element_names)):
                log.warning("Repetition found in `element_names` -> renaming")
                old_element_names = element_names
                element_names = []
                counters = {nn: 0 for nn in old_element_names}
                for nn in old_element_names:
                    if counters[nn] > 0:
                        new_nn = nn + '_'+  str(counters[nn])
                    else:
                        new_nn = nn
                    counters[nn] += 1
                    element_names.append(new_nn)

            assert len(element_names) == len(elements), (
                "`elements` and `element_names` should have the same length"
            )
            element_dict = dict(zip(element_names, elements))

        self.element_dict=element_dict.copy() # avoid modifications if user provided
        self.element_names=list(element_names).copy()

        self.particle_ref = particle_ref

        self._var_management = None
        self._needs_rng = False
        self.tracker = None

    def build_tracker(self, **kwargs):
        assert self.tracker is None, 'The line already has an associated tracker'
        import xtrack as xt # avoid circular import
        self.tracker = xt.Tracker(line=self, **kwargs)
        return self.tracker


    @property
    def elements(self):
        return tuple([self.element_dict[nn] for nn in self.element_names])

    def __getitem__(self, ii):
        if isinstance(ii, str):
            return self.element_dict.__getitem__(ii)
        else:
            names = self.element_names.__getitem__(ii)
            if isinstance(names, str):
                return self.element_dict.__getitem__(names)
            else:
                return [self.element_dict[nn] for nn in names]

    def filter_elements(self, mask=None, exclude_types_starting_with=None):

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

        return new_line

    def cycle(self, index_first_element=None, name_first_element=None):

        if ((index_first_element is not None and name_first_element is not None)
               or (index_first_element is None and name_first_element is None)):
             raise ValueError(
                "Plaese provide either `index_first_element` or `name_first_element`.")

        if name_first_element is not None:
            assert self.element_names.count(name_first_element) == 1, (
                f"name_first_element={name_first_element} occurs more than once!"
            )
            index_first_element = self.element_names.index(name_first_element)

        new_elements = (list(self.elements[index_first_element:])
                        + list(self.elements[:index_first_element]))
        new_element_names = (list(self.element_names[index_first_element:])
                        + list(self.element_names[:index_first_element]))

        return self.__class__(
                         elements=new_elements, element_names=new_element_names)

    def configure_radiation(self, mode=None):
        assert mode in [None, 'mean', 'quantum']
        if mode == 'mean':
            radiation_flag = 1
        elif mode == 'quantum':
            radiation_flag = 2
        else:
            radiation_flag = 0

        for kk, ee in self.element_dict.items():
            if hasattr(ee, 'radiation_flag'):
                ee.radiation_flag = radiation_flag

        if radiation_flag == 2:
            self._needs_rng = True
        else:
            self._needs_rng = False

    def _freeze(self):
        self.element_names = tuple(self.element_names)

    def unfreeze(self):
        self.element_names = list(self.element_names)
        if hasattr(self, 'tracker') and self.tracker is not None:
            self.tracker._invalidate()

    def _frozen_check(self):
        if isinstance(self.element_names, tuple):
            raise ValueError(
                'This action is not allowed as the line is frozen!')

    def __len__(self):
        return len(self.element_names)

    def copy(self, _context=None, _buffer=None):

        elements = {nn: ee.copy(_context=_context, _buffer=_buffer)
                                    for nn, ee in self.element_dict.items()}
        element_names = [nn for nn in self.element_names]

        out = self.__class__(elements=elements, element_names=element_names)

        if self.particle_ref is not None:
            out.particle_ref = self.particle_ref.copy(
                                        _context=_context, _buffer=_buffer)

        if self._var_management is not None:
            out._init_var_management(dct=self._var_management_to_dict())

        return out


    def _var_management_to_dict(self):
        out = {}
        out['_var_management_data'] = deepcopy(self._var_management['data'])
        out['_var_manager'] = self._var_management['manager'].dump()
        return out

    def to_dict(self):
        out = {}
        out["elements"] = {k: el.to_dict() for k, el in self.element_dict.items()}
        out["element_names"] = self.element_names[:]
        if self.particle_ref is not None:
            out['particle_ref'] = self.particle_ref.to_dict()
        if self._var_management is not None:
            out.update(self._var_management_to_dict())
        return out

    def to_pandas(self):
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

    def insert_element(self, index=None, element=None, name=None, at_s=None,
                       s_tol=1e-6):

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
                    not e_to_replace.__class__.__name__.startswith('Limit')):
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
        self._frozen_check()
        assert name not in self.element_dict.keys()
        self.element_dict[name] = element
        self.element_names.append(name)
        return self

    def get_length(self):
        ll = 0
        for ee in self.elements:
            if _is_thick(ee):
                ll += ee.length

        return ll

    def get_s_elements(self, mode="upstream"):
        return self.get_s_position(mode=mode)

    def get_s_position(self, at_elements=None, mode="upstream"):

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

    def remove_inactive_multipoles(self, inplace=True):

        self._frozen_check()

        if not inplace:
            raise NotImplementedError

        newline = Line(elements=[], element_names=[])

        for ee, nn in zip(self.elements, self.element_names):
            if isinstance(ee, (beam_elements.Multipole)):
                aux = [ee.hxl, ee.hyl] + list(ee.knl) + list(ee.ksl)
                if np.sum(np.abs(np.array(aux))) == 0.0:
                    continue
            newline.append_element(ee, nn)


        self.element_names = newline.element_names
        return self

    def remove_zero_length_drifts(self, inplace=True):

        self._frozen_check()

        if not inplace:
            raise NotImplementedError

        newline = Line(elements=[], element_names=[])

        for ee, nn in zip(self.elements, self.element_names):
            if _is_drift(ee):
                if ee.length == 0.0:
                    continue
            newline.append_element(ee, nn)

        self.element_names = newline.element_names
        return self

    def merge_consecutive_drifts(self, inplace=True):

        self._frozen_check()

        if not inplace:
            raise NotImplementedError

        newline = Line(elements=[], element_names=[])

        for ii, (ee, nn) in enumerate(zip(self.elements, self.element_names)):
            if ii == 0:
                newline.append_element(ee, nn)
                continue

            if _is_drift(ee):
                prev_nn = newline.element_names[-1]
                prev_ee = newline.element_dict[prev_nn]
                if _is_drift(prev_ee):
                    prev_ee.length += ee.length
                else:
                    newline.append_element(ee, nn)
            else:
                newline.append_element(ee, nn)

        self.element_dict.update(newline.element_dict)
        self.element_names = newline.element_names
        return self

    def merge_consecutive_multipoles(self, inplace=True):

        self._frozen_check()
        if self._var_management is not None:
            raise NotImplementedError('`merge_consecutive_multipoles` not'
                                      ' available when deferred expressions are'
                                      ' used')

        if not inplace:
            raise NotImplementedError

        newline = Line(elements=[], element_names=[])

        for ee, nn in zip(self.elements, self.element_names):
            if len(newline.elements) == 0:
                newline.append_element(ee, nn)
                continue

            if isinstance(ee, beam_elements.Multipole):
                prev_ee = newline.elements[-1]
                prev_nn = newline.element_names[-1]
                if (isinstance(prev_ee, beam_elements.Multipole)
                    and prev_ee.hxl==ee.hxl==0 and prev_ee.hyl==ee.hyl==0
                    ):

                    oo=max(len(prev_ee.knl), len(prev_ee.ksl),
                           len(ee.knl), len(ee.ksl))
                    knl=np.zeros(oo,dtype=float)
                    ksl=np.zeros(oo,dtype=float)
                    for ii,kk in enumerate(prev_ee.knl):
                        knl[ii]+=kk
                    for ii,kk in enumerate(ee.knl):
                        knl[ii]+=kk
                    for ii,kk in enumerate(prev_ee.ksl):
                        ksl[ii]+=kk
                    for ii,kk in enumerate(ee.ksl):
                        ksl[ii]+=kk
                    newee = beam_elements.Multipole(
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

        self.element_dict.update(newline.element_dict)
        self.element_names = newline.element_names
        return self

    def get_elements_of_type(self, types):
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

    def check_aperture(self):

        elements_df = self.to_pandas()

        elements_df['is_aperture'] = elements_df.element_type.map(lambda s: s.startswith('Limit'))
        elements_df['i_aperture_upstream'] = np.nan
        elements_df['s_aperture_upstream'] = np.nan
        elements_df['i_aperture_downstream'] = np.nan
        elements_df['s_aperture_downstream'] = np.nan

        num_elements = len(self.element_names)

        i_prev_aperture = elements_df[elements_df['is_aperture']].index[0]
        i_next_aperture = 0

        for iee in range(i_prev_aperture, num_elements):

            if iee % 100 == 0:
                print(
                    f'Checking aperture: {round(iee/num_elements*100):2d}%  ',
                    end="\r", flush=True)

            if elements_df.loc[iee, 'element_type'] == 'Drift':
                continue

            if elements_df.loc[iee, 'element_type'] == 'XYShift':
                continue

            if elements_df.loc[iee, 'element_type'] == 'SRotation':
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

        print('Done checking aperture.           ')

        # Identify issues with apertures associate with thin elements
        df_thin_missing_aper = elements_df[elements_df['misses_aperture_upstream'] & ~elements_df['isthick']]
        print(f'{len(df_thin_missing_aper)} thin elements miss associated aperture (upstream):')
        pp(list(df_thin_missing_aper.name))

        # Identify issues with apertures associate with thin elements
        df_thick_missing_aper = elements_df[
            (elements_df['misses_aperture_upstream'] | elements_df['misses_aperture_downstream'])
            & elements_df['isthick']]
        print(f'{len(df_thick_missing_aper)} thick elements miss associated aperture (upstream or downstream):')
        pp(list(df_thick_missing_aper.name))

        return elements_df


mathfunctions = type('math', (), {})
mathfunctions.sqrt=math.sqrt
mathfunctions.log=math.log
mathfunctions.log10=math.log10
mathfunctions.exp=math.exp
mathfunctions.sin=math.sin
mathfunctions.cos=math.cos
mathfunctions.tan=math.tan
mathfunctions.asin=math.asin
mathfunctions.acos=math.acos
mathfunctions.atan=math.atan
mathfunctions.sinh=math.sinh
mathfunctions.cosh=math.cosh
mathfunctions.tanh=math.tanh
mathfunctions.sinc=np.sinc
mathfunctions.abs=math.fabs
mathfunctions.erf=math.erf
mathfunctions.erfc=math.erfc
mathfunctions.floor=math.floor
mathfunctions.ceil=math.ceil
mathfunctions.round=np.round
mathfunctions.frac=lambda x: (x%1)

def _deserialize_element(el, class_dict, _buffer):
    eldct = el.copy()
    eltype = class_dict[eldct.pop('__class__')]
    if hasattr(eltype, '_XoStruct'):
        return eltype.from_dict(eldct, _buffer=_buffer)
    else:
        return eltype.from_dict(eldct)
