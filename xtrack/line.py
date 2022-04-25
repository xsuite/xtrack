import json
import math
import logging
from copy import deepcopy

import numpy as np

import xobjects as xo
import xpart as xp

from .loader_sixtrack import _expand_struct
from .loader_mad import madx_sequence_to_xtrack_line
from .beam_elements import element_classes, Multipole
from . import beam_elements
from .beam_elements import Drift

log=logging.getLogger(__name__)

def mk_class_namespace(extra_classes):
    try:
       import xfields as xf
       all_classes= element_classes + xf.element_classes + extra_classes
    except ImportError:
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




# missing access to particles._m:
deg2rad = np.pi / 180.

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self



class Line:

    @classmethod
    def from_dict(cls, dct, _context=None, _buffer=None, classes=()):
        class_dict=mk_class_namespace(classes)

        _buffer, _ =xo.get_a_buffer(size=8,context=_context, buffer=_buffer)
        elements = []
        for el in dct["elements"]:
            eltype = class_dict[el["__class__"]]
            eldct=el.copy()
            del eldct['__class__']
            if hasattr(eltype,'XoStruct'):
               newel = eltype.from_dict(eldct,_buffer=_buffer)
            else:
               newel = eltype.from_dict(eldct)
            elements.append(newel)

        self = cls(elements=elements, element_names=dct['element_names'])

        if 'particle_ref' in dct.keys():
            self.particle_ref = xp.Particles.from_dict(dct['particle_ref'],
                                    _context=_buffer.context)

        if '_var_manager' in dct.keys():
            self._init_var_management()
            manager = self._var_management['manager']
            for kk in self._var_management['data'].keys():
                self._var_management['data'][kk].update(
                                            dct['_var_management_data'][kk])
            manager.load(dct['_var_manager'])

        return self

    @classmethod
    def from_sixinput(cls, sixinput, classes=()):
        class_dict=mk_class_namespace(classes)

        line_data, rest, iconv = _expand_struct(sixinput, convert=class_dict)

        ele_names = [dd[0] for dd in line_data]
        elements = [dd[2] for dd in line_data]

        line = cls(elements=elements, element_names=ele_names)

        other_info = {}
        other_info["rest"] = rest
        other_info["iconv"] = iconv

        line.other_info = other_info

        return line

    @classmethod
    def from_madx_sequence(
        cls,
        sequence,
        classes=(),
        ignored_madtypes=[],
        exact_drift=False,
        drift_threshold=1e-6,
        deferred_expressions=False,
        install_apertures=False,
        apply_madx_errors=False,
    ):

        class_dict=mk_class_namespace(classes)

        line = madx_sequence_to_xtrack_line(
            sequence,
            class_dict,
            ignored_madtypes=ignored_madtypes,
            exact_drift=exact_drift,
            drift_threshold=drift_threshold,
            install_apertures=install_apertures,
            deferred_expressions=deferred_expressions)

        if apply_madx_errors:
            line._apply_madx_errors(sequence)

        return line

    def _init_var_management(self):

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
                    new_elements.append(Drift(length==ee.length))
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

    def _frozen_check(self):
        if isinstance(self.element_names, tuple):
            raise ValueError(
                'This action is not allowed as the line is frozen!')

    def __len__(self):
        return len(self.element_names)

    def to_dict(self):
        out = {}
        out["elements"] = [el.to_dict() for el in self.elements]
        out["element_names"] = self.element_names[:]
        if self.particle_ref is not None:
            out['particle_ref'] = self.particle_ref.to_dict()
        if self._var_management is not None:
            out['_var_management_data'] = deepcopy(self._var_management['data'])
            out['_var_manager'] = self._var_management['manager'].dump()
        return out

    def copy(self):
        return self.__class__.from_dict(self.to_dict())

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

    def remove_inactive_multipoles(self, inplace=False):

        self._frozen_check()

        newline = Line(elements=[], element_names=[])

        for ee, nn in zip(self.elements, self.element_names):
            if isinstance(ee, (beam_elements.Multipole)):
                aux = [ee.hxl, ee.hyl] + list(ee.knl) + list(ee.ksl)
                if np.sum(np.abs(np.array(aux))) == 0.0:
                    continue
            newline.append_element(ee, nn)

        if inplace:
            self.element_names = newline.element_names
            return self
        else:
            return newline

    def remove_zero_length_drifts(self, inplace=False):

        self._frozen_check()

        newline = Line(elements=[], element_names=[])

        for ee, nn in zip(self.elements, self.element_names):
            if _is_drift(ee):
                if ee.length == 0.0:
                    continue
            newline.append_element(ee, nn)

        if inplace:
            self.element_names = newline.element_names
            return self
        else:
            return newline

    def merge_consecutive_drifts(self, inplace=False):

        self._frozen_check()

        newline = Line(elements=[], element_names=[])

        for ee, nn in zip(self.elements, self.element_names):
            if len(newline.elements) == 0:
                newline.append_element(ee, nn)
                continue

            if _is_drift(ee):
                prev_nn = newline.element_names[-1]
                prev_ee = newline.element_dict[prev_nn]
                if _is_drift(prev_ee):
                    prev_ee.length += ee.length
                    newline.element_names[-1] = prev_nn
                else:
                    newline.append_element(ee, nn)
            else:
                newline.append_element(ee, nn)

        if inplace:
            self.element_dict.update(newline.element_dict)
            self.element_names = newline.element_names
            return self
        else:
            return newline

    def merge_consecutive_multipoles(self, inplace=False):

        self._frozen_check()
        if self._var_management is not None:
            raise NotImplementedError('`merge_consecutive_multipoles` not'
                                      ' available when deferred expressions are'
                                      ' used')

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

        if inplace:
            self.element_dict.update(newline.element_dict)
            self.element_names = newline.element_names
            return self
        else:
            return newline

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

    def _find_element_ids(self, element_name):
        """Find element_name in this Line instance's
        self.elements_name list. Assumes the names are unique.

        Return index before and after the element, taking into account
        attached _aperture instances (LimitRect, LimitEllipse, ...)
        which would follow the element occurrence in the list.

        Raises IndexError if element_name not found in this Line.
        """
        # will raise error if element not present:
        idx_el = self.element_names.index(element_name)
        try:
            # if aperture marker is present
            idx_after_el = self.element_names.index(element_name + "_aperture") + 1
        except ValueError:
            # if aperture marker is not present
            idx_after_el = idx_el + 1
        return idx_el, idx_after_el

    def _add_offset_error_to(self, element_name, dx=0, dy=0):
        idx_el, idx_after_el = self._find_element_ids(element_name)
        xyshift = beam_elements.XYShift(dx=dx, dy=dy)
        inv_xyshift = beam_elements.XYShift(dx=-dx, dy=-dy)
        self.insert_element(idx_el, xyshift, element_name + "_offset_in")
        self.insert_element(
            idx_after_el + 1, inv_xyshift, element_name + "_offset_out"
        )

    def _add_aperture_offset_error_to(self, element_name, arex=0, arey=0):
        idx_el, idx_after_el = self._find_element_ids(element_name)
        idx_el_aper = idx_after_el - 1
        if not self.element_names[idx_el_aper] == element_name + "_aperture":
            # it is allowed to provide arex/arey without providing an aperture
            print('Info: Element', element_name, ': arex/y provided without aperture -> arex/y ignored')
            return
        xyshift = beam_elements.XYShift(dx=arex, dy=arey)
        inv_xyshift = beam_elements.XYShift(dx=-arex, dy=-arey)
        self.insert_element(idx_el_aper, xyshift, element_name + "_aperture_offset_in")
        self.insert_element(
            idx_after_el + 1, inv_xyshift, element_name + "_aperture_offset_out"
        )

    def _add_tilt_error_to(self, element_name, angle):
        '''Alignment error of transverse rotation around s-axis.
        The element corresponding to the given `element_name`
        gets wrapped by SRotation elements with rotation angle
        `angle`.

        In the case of a thin dipole component, the corresponding
        curvature terms in the Multipole (hxl and hyl) are rotated
        by `angle` as well.
        '''
        idx_el, idx_after_el = self._find_element_ids(element_name)
        element = self.elements[self.element_names.index(element_name)]
        if isinstance(element, beam_elements.Multipole) and (
                element.hxl or element.hyl):
            dpsi = angle * deg2rad

            hxl0 = element.hxl
            hyl0 = element.hyl

            hxl1 = hxl0 * np.cos(dpsi) - hyl0 * np.sin(dpsi)
            hyl1 = hxl0 * np.sin(dpsi) + hyl0 * np.cos(dpsi)

            element.hxl = hxl1
            element.hyl = hyl1
        srot = beam_elements.SRotation(angle=angle)
        inv_srot = beam_elements.SRotation(angle=-angle)
        self.insert_element(idx_el, srot, element_name + "_tilt_in")
        self.insert_element(idx_after_el + 1, inv_srot, element_name + "_tilt_out")

    def _add_multipole_error_to(self, element_name, knl=[], ksl=[]):
        # will raise error if element not present:
        assert element_name in self.element_names
        element_index = self.element_names.index(element_name)
        element = self.elements[element_index]

        new_order = max([len(knl), len(ksl), len(element.knl), len(element.ksl)])
        new_knl = new_order*[0]
        new_ksl = new_order*[0]

        # Original strengths
        for ii, vv in enumerate(element.knl):
            new_knl[ii] += element.knl[ii]
        for ii, vv in enumerate(element.ksl):
            new_ksl[ii] += element.ksl[ii]

        new_element = Multipole(knl=new_knl, ksl=new_ksl,
                length=element.length, hxl=element.hxl,
                hyl=element.hyl, radiation_flag=element.radiation_flag)

        self.element_dict[element_name] = new_element

        # Errors
        if self._var_management is not None:
            # Handle deferred expressions
            lref = self._var_management['lref']
            for ii, vv in enumerate(knl):
                lref[element_name].knl[ii] += knl[ii]
            for ii, vv in enumerate(ksl):
                lref[element_name].ksl[ii] += ksl[ii]
        else:
            for ii, vv in enumerate(knl):
                new_element.knl[ii] += knl[ii]
            for ii, vv in enumerate(ksl):
                new_element.ksl[ii] += ksl[ii]

    def _apply_madx_errors(self, madx_sequence):
        """Applies errors from MAD-X sequence to existing
        elements in this Line instance.

        Return names of MAD-X elements with existing align_errors
        or field_errors which were not found in the elements of
        this Line instance (and thus not treated).

        Example via cpymad:
            madx = cpymad.madx.Madx()

            # (...set up lattice and errors in cpymad...)

            seq = madx.sequence.some_lattice
            line = Line.from_madx_sequence(
                                    seq,
                                    apply_madx_errors=True
                              )
        """
        elements_not_found = []
        for element, element_name in zip(
                madx_sequence.expanded_elements,
                madx_sequence.expanded_element_names()
        ):
            if element_name not in self.element_names:
                if element.align_errors or element.field_errors:
                    elements_not_found.append(element_name)
                    continue

            if element.align_errors:
                # add offset
                dx = element.align_errors.dx
                dy = element.align_errors.dy
                if dx or dy:
                    self._add_offset_error_to(element_name, dx, dy)

                # add tilt
                dpsi = element.align_errors.dpsi
                if dpsi:
                    self._add_tilt_error_to(element_name, angle=dpsi / deg2rad)

                # add aperture-only offset
                arex = element.align_errors.arex
                arey = element.align_errors.arey
                if arex or arey:
                    self._add_aperture_offset_error_to(element_name, arex, arey)

                # check for errors which cannot be treated yet:
                #for error_type in dir(element.align_errors):
                 #   if not error_type[0] == '_' and \
                  #          error_type not in ['dx', 'dy', 'dpsi', 'arex',
                   #                            'arey', 'count', 'index']:
                        #print(
                        #    f'Warning: MAD-X error type "{error_type}"'
                        #    " not implemented yet."
                        #)

            if element.field_errors:
                # add multipole error
                if any(element.field_errors.dkn) or \
                            any(element.field_errors.dks):
                    knl = element.field_errors.dkn
                    ksl = element.field_errors.dks
                    on=np.where(knl)[0]
                    os=np.where(ksl)[0]
                    on = on[-1] if len(on)>0 else 0
                    os = os[-1] if len(os)>0 else 0
                    oo = max(os,on)+1
                    knl = knl[:oo]  # delete trailing zeros
                    ksl = ksl[:oo]  # to keep order low
                    self._add_multipole_error_to(element_name, knl, ksl)

        return elements_not_found

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
