import xobjects as xo
import numpy as np

from ..general import _pkg_root
from ..base_element import BeamElement
from .slice_base import (
    _SliceBase, COMMON_SLICE_XO_FIELDS,
    _raise_if_parent_has_transverse_rotation,
)
from .elements import (
    Bend, Quadrupole, Sextupole,
    Octupole, RBend, UniformSolenoid, DipoleEdge, Marker, MultipoleEdge
)
from ..survey import advance_element as survey_advance_element

def _parent_total_kn_ks(parent):
    if parent.length == 0:
        raise ValueError(
            'Equivalent edge elements need a parent with non-zero length.')
    knl, ksl = parent.get_total_knl_ksl()
    return knl / parent.length, ksl / parent.length

def _parent_total_kn_ks_for_multipole_edge(parent):
    kn, ks = _parent_total_kn_ks(parent)
    kn[0] = 0
    ks[0] = 0
    return kn, ks


class _ThinSliceEdgeBase(_SliceBase):

    rot_and_shift_from_parent = True
    allow_loss_refinement = False
    isthick=False
    _inherit_strengths = False

class ThinSliceBendEntry(_ThinSliceEdgeBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Bend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_dipole_edge_nonlinear.h'),
        '#include "xtrack/beam_elements/elements_src/thin_slice_bend_entry.h"'
    ]

    def get_equivalent_element(self):

        _raise_if_parent_has_transverse_rotation(self._parent)

        if self._parent.edge_entry_active:
            kn, _ = _parent_total_kn_ks(self._parent)
            return DipoleEdge(
                k=kn[0],
                e1=self._parent.edge_entry_angle,
                e1_fd=self._parent.edge_entry_angle_fdown,
                hgap=self._parent.edge_entry_hgap,
                fint=self._parent.edge_entry_fint,
                model=self._parent.edge_entry_model,
                side='entry',
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)

class ThinSliceBendExit(_ThinSliceEdgeBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Bend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_bend_exit.h"'
    ]

    def get_equivalent_element(self):

        _raise_if_parent_has_transverse_rotation(self._parent)

        if self._parent.edge_exit_active:
            kn, _ = _parent_total_kn_ks(self._parent)
            return DipoleEdge(
                k=kn[0],
                e1=self._parent.edge_exit_angle,
                e1_fd=self._parent.edge_exit_angle_fdown,
                hgap=self._parent.edge_exit_hgap,
                fint=self._parent.edge_exit_fint,
                model=self._parent.edge_exit_model,
                delta_taper=self.delta_taper,
                side='exit',
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)

class ThinSliceQuadrupoleEntry(_ThinSliceEdgeBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Quadrupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_quadrupole_entry.h"'
    ]

    def get_equivalent_element(self):
        _raise_if_parent_has_transverse_rotation(self._parent)

        if self._parent.edge_entry_active:
            kn, ks = _parent_total_kn_ks_for_multipole_edge(self._parent)
            return MultipoleEdge(
                kn=kn,
                ks=ks,
                is_exit=False,
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)

class ThinSliceQuadrupoleExit(_ThinSliceEdgeBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Quadrupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_quadrupole_exit.h"'
    ]

    def get_equivalent_element(self):
        _raise_if_parent_has_transverse_rotation(self._parent)

        if self._parent.edge_exit_active:
            kn, ks = _parent_total_kn_ks_for_multipole_edge(self._parent)
            return MultipoleEdge(
                kn=kn,
                ks=ks,
                is_exit=True,
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)

class ThinSliceSextupoleEntry(_ThinSliceEdgeBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Sextupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_sextupole_entry.h"'
    ]

    def get_equivalent_element(self):
        _raise_if_parent_has_transverse_rotation(self._parent)

        if self._parent.edge_entry_active:
            kn, ks = _parent_total_kn_ks_for_multipole_edge(self._parent)
            return MultipoleEdge(
                kn=kn,
                ks=ks,
                is_exit=False,
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)

class ThinSliceSextupoleExit(_ThinSliceEdgeBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Sextupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_sextupole_exit.h"'
    ]

    def get_equivalent_element(self):
        _raise_if_parent_has_transverse_rotation(self._parent)

        if self._parent.edge_exit_active:
            kn, ks = _parent_total_kn_ks_for_multipole_edge(self._parent)
            return MultipoleEdge(
                kn=kn,
                ks=ks,
                is_exit=True,
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)

class ThinSliceOctupoleEntry(_ThinSliceEdgeBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Octupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_octupole_entry.h"'
    ]

    def get_equivalent_element(self):
        _raise_if_parent_has_transverse_rotation(self._parent)

        if self._parent.edge_entry_active:
            kn, ks = _parent_total_kn_ks_for_multipole_edge(self._parent)
            return MultipoleEdge(
                kn=kn,
                ks=ks,
                is_exit=False,
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)

class ThinSliceOctupoleExit(_ThinSliceEdgeBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Octupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_octupole_exit.h"'
    ]

    def get_equivalent_element(self):
        _raise_if_parent_has_transverse_rotation(self._parent)

        if self._parent.edge_exit_active:
            kn, ks = _parent_total_kn_ks_for_multipole_edge(self._parent)
            return MultipoleEdge(
                kn=kn,
                ks=ks,
                is_exit=True,
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)

class ThinSliceUniformSolenoidEntry(_ThinSliceEdgeBase, BeamElement):

    _xofields = {'_parent': xo.Ref(UniformSolenoid), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_uniform_solenoid_entry.h"'
    ]


class ThinSliceUniformSolenoidExit(_ThinSliceEdgeBase, BeamElement):

    _xofields = {'_parent': xo.Ref(UniformSolenoid), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_uniform_solenoid_exit.h"'
    ]

class ThinSliceRBendEntry(_ThinSliceEdgeBase, BeamElement):

    _xofields = {'_parent': xo.Ref(RBend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_rbend_entry.h"'
    ]

    def get_equivalent_element(self):

        _raise_if_parent_has_transverse_rotation(self._parent)

        if self._parent.rbend_model == "straight-body":
            return self # No replacement possible (not yet supported), element
                        # left where it is

        if self._parent.edge_entry_active:
            kn, _ = _parent_total_kn_ks(self._parent)
            return DipoleEdge(
                k=kn[0],
                e1=self._parent.edge_entry_angle + self._parent.angle / 2,
                e1_fd=self._parent.edge_entry_angle_fdown,
                hgap=self._parent.edge_entry_hgap,
                fint=self._parent.edge_entry_fint,
                model=self._parent.edge_entry_model,
                side='entry',
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)

    def _propagate_survey(self, v, w, backtrack):

        if self._parent.rbend_model == "straight-body":
            if backtrack:
                if abs(self._parent.angle) > 1e-10:  # avoid numerical issues
                    v, w = survey_advance_element(
                        v               = v,
                        w               = w,
                        length          = 0,
                        angle           = 0,
                        tilt            = 0,
                        ref_shift_x     = self._parent._x0_in * np.cos(self._parent.rot_s_rad),
                        ref_shift_y     = self._parent._x0_in * np.sin(self._parent.rot_s_rad),
                        ref_rot_x_rad   = 0,
                        ref_rot_y_rad   = 0,
                        ref_rot_s_rad   = 0,
                    )
                v, w = survey_advance_element(
                    v               = v,
                    w               = w,
                    length          = 0,
                    angle           = -self._parent._angle_in,
                    tilt            = self._parent.rot_s_rad,
                    ref_shift_x     = 0,
                    ref_shift_y     = 0,
                    ref_rot_x_rad   = 0,
                    ref_rot_y_rad   = 0,
                    ref_rot_s_rad   = 0,
                )
            else:
                v, w = survey_advance_element(
                    v               = v,
                    w               = w,
                    length          = 0,
                    angle           = self._parent._angle_in,
                    tilt            = self._parent.rot_s_rad,
                    ref_shift_x     = 0,
                    ref_shift_y     = 0,
                    ref_rot_x_rad   = 0,
                    ref_rot_y_rad   = 0,
                    ref_rot_s_rad   = 0,
                )
                if abs(self._parent.angle) > 1e-10:  # avoid numerical issues
                    v, w = survey_advance_element(
                        v               = v,
                        w               = w,
                        length          = 0,
                        angle           = 0,
                        tilt            = 0,
                        ref_shift_x     = -self._parent._x0_in * np.cos(self._parent.rot_s_rad),
                        ref_shift_y     = -self._parent._x0_in * np.sin(self._parent.rot_s_rad),
                        ref_rot_x_rad   = 0,
                        ref_rot_y_rad   = 0,
                        ref_rot_s_rad   = 0,
                    )
        return v, w

class ThinSliceRBendExit(_ThinSliceEdgeBase, BeamElement):

    _xofields = {'_parent': xo.Ref(RBend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_rbend_exit.h"'
    ]

    def get_equivalent_element(self):

        _raise_if_parent_has_transverse_rotation(self._parent)

        if self._parent.rbend_model == "straight-body":
            return self # No replacement possible (not yet supported), element
                        # left where it is

        if self._parent.edge_exit_active:
            kn, _ = _parent_total_kn_ks(self._parent)
            return DipoleEdge(
                k=kn[0],
                e1=self._parent.edge_exit_angle + self._parent.angle / 2,
                e1_fd=self._parent.edge_exit_angle_fdown,
                hgap=self._parent.edge_exit_hgap,
                fint=self._parent.edge_exit_fint,
                model=self._parent.edge_exit_model,
                side='exit',
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)

    def _propagate_survey(self, v, w, backtrack):

        if self._parent.rbend_model == "straight-body":
            if backtrack:
                v, w = survey_advance_element(
                    v               = v,
                    w               = w,
                    length          = 0,
                    angle           = -self._parent._angle_out,
                    tilt            = self._parent.rot_s_rad,
                    ref_shift_x     = 0,
                    ref_shift_y     = 0,
                    ref_rot_x_rad   = 0,
                    ref_rot_y_rad   = 0,
                    ref_rot_s_rad   = 0,
                )
                if abs(self._parent.angle) > 1e-10:  # avoid numerical issues
                    v, w = survey_advance_element(
                        v               = v,
                        w               = w,
                        length          = 0,
                        angle           = 0,
                        tilt            = 0,
                        ref_shift_x     = -self._parent._x0_out * np.cos(self._parent.rot_s_rad),
                        ref_shift_y     = -self._parent._x0_out * np.sin(self._parent.rot_s_rad),
                        ref_rot_x_rad   = 0,
                        ref_rot_y_rad   = 0,
                        ref_rot_s_rad   = 0,
                    )
            else:
                if abs(self._parent.angle) > 1e-10:  # avoid numerical issues
                    v, w = survey_advance_element(
                        v               = v,
                        w               = w,
                        length          = 0,
                        angle           = 0,
                        tilt            = 0,
                        ref_shift_x     = self._parent._x0_out * np.cos(self._parent.rot_s_rad),
                        ref_shift_y     = self._parent._x0_out * np.sin(self._parent.rot_s_rad),
                        ref_rot_x_rad   = 0,
                        ref_rot_y_rad   = 0,
                        ref_rot_s_rad   = 0,
                    )
                v, w = survey_advance_element(
                    v               = v,
                    w               = w,
                    length          = 0,
                    angle           = self._parent._angle_out,
                    tilt            = self._parent.rot_s_rad,
                    ref_shift_x     = 0,
                    ref_shift_y     = 0,
                    ref_rot_x_rad   = 0,
                    ref_rot_y_rad   = 0,
                    ref_rot_s_rad   = 0,
                )
        return v, w
