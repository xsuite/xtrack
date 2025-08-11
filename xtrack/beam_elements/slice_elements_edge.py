import xobjects as xo
import numpy as np

from ..general import _pkg_root
from ..base_element import BeamElement
from .slice_base import _SliceBase, COMMON_SLICE_XO_FIELDS
from .elements import (
    Bend, Quadrupole, Sextupole,
    Octupole, RBend, UniformSolenoid, DipoleEdge, Marker, MultipoleEdge
)
from ..survey import advance_element as survey_advance_element

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
        '#include <beam_elements/elements_src/thin_slice_bend_entry.h>'
    ]

    def get_equivalent_element(self):

        if self._parent.edge_entry_active:
            return DipoleEdge(
                k=self._parent.k0,
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
        '#include <beam_elements/elements_src/thin_slice_bend_exit.h>'
    ]

    def get_equivalent_element(self):

        if self._parent.edge_exit_active:
            return DipoleEdge(
                k=self._parent.k0,
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
        '#include <beam_elements/elements_src/thin_slice_quadrupole_entry.h>'
    ]

    def get_equivalent_element(self):
        if self._parent.edge_entry_active:
            return MultipoleEdge(
                kn=[0, self._parent.k1],
                ks=[0, self._parent.k1s],
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
        '#include <beam_elements/elements_src/thin_slice_quadrupole_exit.h>'
    ]

    def get_equivalent_element(self):
        if self._parent.edge_exit_active:
            return MultipoleEdge(
                kn=[0, self._parent.k1],
                ks=[0, self._parent.k1s],
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
        '#include <beam_elements/elements_src/thin_slice_sextupole_entry.h>'
    ]

    def get_equivalent_element(self):
        if self._parent.edge_entry_active:
            return MultipoleEdge(
                kn=[0, 0, self._parent.k2],
                ks=[0, 0, self._parent.k2s],
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
        '#include <beam_elements/elements_src/thin_slice_sextupole_exit.h>'
    ]

    def get_equivalent_element(self):
        if self._parent.edge_exit_active:
            return MultipoleEdge(
                kn=[0, 0, self._parent.k2],
                ks=[0, 0, self._parent.k2s],
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
        '#include <beam_elements/elements_src/thin_slice_octupole_entry.h>'
    ]

    def get_equivalent_element(self):
        if self._parent.edge_entry_active:
            return MultipoleEdge(
                kn=[0, 0, 0, self._parent.k3],
                ks=[0, 0, 0, self._parent.k3s],
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
        '#include <beam_elements/elements_src/thin_slice_octupole_exit.h>'
    ]

    def get_equivalent_element(self):
        if self._parent.edge_exit_active:
            return MultipoleEdge(
                kn=[0, 0, 0, self._parent.k3],
                ks=[0, 0, 0, self._parent.k3s],
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
        '#include <beam_elements/elements_src/thin_slice_uniform_solenoid_entry.h>'
    ]

    def get_equivalent_element(self):
        if self._parent.edge_entry_active:
            raise NotImplementedError
        else:
            return Marker(_buffer=self._buffer)

class ThinSliceUniformSolenoidExit(_ThinSliceEdgeBase, BeamElement):

    _xofields = {'_parent': xo.Ref(UniformSolenoid), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/thin_slice_uniform_solenoid_exit.h>'
    ]

    def get_equivalent_element(self):
        if self._parent.edge_entry_active:
            raise NotImplementedError
        else:
            return Marker(_buffer=self._buffer)

class ThinSliceRBendEntry(_ThinSliceEdgeBase, BeamElement):

    _xofields = {'_parent': xo.Ref(RBend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/thin_slice_rbend_entry.h>'
    ]

    def get_equivalent_element(self):

        if self._parent.edge_entry_active:
            return DipoleEdge(
                k=self._parent.k0,
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
            rbend_shift_tot = self._parent.sagitta / 2  + self._parent.rbend_shift
            if backtrack:
                if abs(self._parent.angle) > 1e-10:  # avoid numerical issues
                    v, w = survey_advance_element(
                        v               = v,
                        w               = w,
                        length          = 0,
                        angle           = 0,
                        tilt            = 0,
                        ref_shift_x     = -rbend_shift_tot * np.cos(self._parent.rot_s_rad),
                        ref_shift_y     = -rbend_shift_tot * np.sin(self._parent.rot_s_rad),
                        ref_rot_x_rad   = 0,
                        ref_rot_y_rad   = 0,
                        ref_rot_s_rad   = 0,
                    )
                v, w = survey_advance_element(
                    v               = v,
                    w               = w,
                    length          = 0,
                    angle           = -self._parent.angle / 2.,
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
                    angle           = self._parent.angle / 2.,
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
                        ref_shift_x     = rbend_shift_tot * np.cos(self._parent.rot_s_rad),
                        ref_shift_y     = rbend_shift_tot * np.sin(self._parent.rot_s_rad),
                        ref_rot_x_rad   = 0,
                        ref_rot_y_rad   = 0,
                        ref_rot_s_rad   = 0,
                    )
        return v, w

class ThinSliceRBendExit(_ThinSliceEdgeBase, BeamElement):

    _xofields = {'_parent': xo.Ref(RBend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/thin_slice_rbend_exit.h>'
    ]

    def get_equivalent_element(self):

        if self._parent.edge_exit_active:
            return DipoleEdge(
                k=self._parent.k0,
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
            rbend_shift_tot = self._parent.sagitta / 2  + self._parent.rbend_shift
            if backtrack:
                v, w = survey_advance_element(
                    v               = v,
                    w               = w,
                    length          = 0,
                    angle           = -self._parent.angle / 2.,
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
                        ref_shift_x     = rbend_shift_tot * np.cos(self._parent.rot_s_rad),
                        ref_shift_y     = rbend_shift_tot * np.sin(self._parent.rot_s_rad),
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
                        ref_shift_x     = -rbend_shift_tot * np.cos(self._parent.rot_s_rad),
                        ref_shift_y     = -rbend_shift_tot * np.sin(self._parent.rot_s_rad),
                        ref_rot_x_rad   = 0,
                        ref_rot_y_rad   = 0,
                        ref_rot_s_rad   = 0,
                    )
                v, w = survey_advance_element(
                    v               = v,
                    w               = w,
                    length          = 0,
                    angle           = self._parent.angle / 2.,
                    tilt            = self._parent.rot_s_rad,
                    ref_shift_x     = 0,
                    ref_shift_y     = 0,
                    ref_rot_x_rad   = 0,
                    ref_rot_y_rad   = 0,
                    ref_rot_s_rad   = 0,
                )
        return v, w

