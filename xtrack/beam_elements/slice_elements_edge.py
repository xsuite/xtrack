import xobjects as xo
import numpy as np

from ..general import _pkg_root
from ..base_element import BeamElement
from .slice_base import (
    _SliceBase, COMMON_SLICE_XO_FIELDS,
    _raise_if_parent_has_transverse_rotation,
)
from .bend import Bend
from .dipole_edge import DipoleEdge
from .magnet_edge import MagnetEdge
from .marker import Marker
from .multipole_edge import MultipoleEdge
from .octupole import Octupole
from .quadrupole import Quadrupole
from .rbend import RBend
from .sextupole import Sextupole
from .uniform_solenoid import UniformSolenoid

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


def _has_qint(parent, face):
    return any(getattr(parent, f'edge_{face}_{qq}') != 0 for qq in ('qint1', 'qint2'))


def _alignment_from_parent(edge_slice):
    parent = edge_slice._parent
    return {
        'shift_x': parent.shift_x,
        'shift_y': parent.shift_y,
        'shift_s': parent.shift_s,
        'rot_s_rad': parent.rot_s_rad,
        '_buffer': edge_slice._buffer,
    }


def _magnet_edge_from_parent(edge_slice, model, **face):
    """MagnetEdge of a face of the parent of an edge slice: same maps as the
    face. The wedges act on the thick strengths only: k0...k3 go to kn, ks,
    the rest of the total strengths to knl, ksl."""
    parent = edge_slice._parent
    k_thick = np.zeros(4)
    ks_thick = np.zeros(4)
    for ii in range(4):
        # The k0 of a bend may read 'from_h': its value is _k0
        name = '_k0' if ii == 0 and hasattr(parent, '_k0') else f'k{ii}'
        k_thick[ii] = getattr(parent, name, 0.0)
        ks_thick[ii] = getattr(parent, f'k{ii}s', 0.0)
    knl, ksl = parent.get_total_knl_ksl()
    knl = np.array(knl, dtype=float)
    ksl = np.array(ksl, dtype=float)
    nn = min(4, len(knl))
    knl[:nn] -= k_thick[:nn] * parent.length
    ksl[:nn] -= ks_thick[:nn] * parent.length
    return MagnetEdge(
        model=model,
        is_exit=edge_slice._face == 'exit',
        kn=k_thick,
        ks=ks_thick,
        knl=knl,
        ksl=ksl,
        length=parent.length,
        **_alignment_from_parent(edge_slice),
        **face,
    )


_DIPOLE_EDGE_MODEL = {
    'linear': 'linear',
    'full': 'full',
    'no-mult': 'full',
    'suppressed': 'suppressed',
}


class _ThinSliceEdgeBase(_SliceBase):

    rot_and_shift_from_parent = True
    allow_loss_refinement = False
    isthick=False
    _inherit_strengths = False

class _ThinSliceDipoleEdgeBase:
    """Edge slice of a bend; ``_face`` is 'entry' or 'exit'."""

    _face = None

    def _face_angle_offset(self):
        return 0.0

    def get_equivalent_element(self):

        _raise_if_parent_has_transverse_rotation(self._parent)

        parent, face = self._parent, self._face
        if not getattr(parent, f'edge_{face}_active'):
            return Marker(_buffer=self._buffer)

        kn, ks = _parent_total_kn_ks(parent)
        extra = {'delta_taper': self.delta_taper} if face == 'exit' else {}
        model = getattr(parent, f'edge_{face}_model')
        has_mult = np.any(kn[1:] != 0) or np.any(ks[1:] != 0)
        if (model == 'full' and has_mult) or (
            model in ('full', 'no-mult') and _has_qint(parent, face)
        ):
            # The DipoleEdge has neither the multipole fringe nor the soft
            # quadrupole fringe
            return _magnet_edge_from_parent(
                self,
                model,
                face_angle=(
                    getattr(parent, f'edge_{face}_angle') + self._face_angle_offset()
                ),
                face_angle_feed_down=getattr(parent, f'edge_{face}_angle_fdown'),
                half_gap=getattr(parent, f'edge_{face}_hgap'),
                fringe_integral=getattr(parent, f'edge_{face}_fint'),
                qint1=getattr(parent, f'edge_{face}_qint1'),
                qint2=getattr(parent, f'edge_{face}_qint2'),
            )
        return DipoleEdge(
            k=kn[0],
            e1=getattr(parent, f'edge_{face}_angle') + self._face_angle_offset(),
            e1_fd=getattr(parent, f'edge_{face}_angle_fdown'),
            hgap=getattr(parent, f'edge_{face}_hgap'),
            fint=getattr(parent, f'edge_{face}_fint'),
            # The DipoleEdge has the dipole fringe only: its "full" model is
            # the "no-mult" one of the bend
            model=_DIPOLE_EDGE_MODEL[model],
            **extra,
            side=face,
            **_alignment_from_parent(self),
        )


class _ThinSliceMultipoleEdgeBase:
    """Edge slice of a straight magnet; ``_face`` is 'entry' or 'exit'."""

    _face = None

    def get_equivalent_element(self):
        _raise_if_parent_has_transverse_rotation(self._parent)

        parent, face = self._parent, self._face
        model = getattr(parent, f'edge_{face}_model')
        if not getattr(parent, f'edge_{face}_active') or model == 'suppressed':
            return Marker(_buffer=self._buffer)

        # The MultipoleEdge has neither the dipole fringe nor the soft
        # quadrupole fringe
        if model == 'no-mult' or _has_qint(parent, face):
            return _magnet_edge_from_parent(
                self,
                model,
                qint1=getattr(parent, f'edge_{face}_qint1'),
                qint2=getattr(parent, f'edge_{face}_qint2'),
            )

        kn, ks = _parent_total_kn_ks_for_multipole_edge(parent)
        return MultipoleEdge(
            kn=kn,
            ks=ks,
            is_exit=face == 'exit',
            **_alignment_from_parent(self),
        )


class ThinSliceBendEntry(_ThinSliceEdgeBase, _ThinSliceDipoleEdgeBase, BeamElement):
    _face = 'entry'

    _xofields = {'_parent': xo.Ref(Bend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_dipole_edge_nonlinear.h'),
        '#include "xtrack/beam_elements/elements_src/thin_slice_bend_entry.h"'
    ]


class ThinSliceBendExit(_ThinSliceEdgeBase, _ThinSliceDipoleEdgeBase, BeamElement):

    _face = 'exit'

    _xofields = {'_parent': xo.Ref(Bend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_bend_exit.h"'
    ]


class ThinSliceQuadrupoleEntry(
    _ThinSliceEdgeBase, _ThinSliceMultipoleEdgeBase, BeamElement
):
    _face = 'entry'

    _xofields = {'_parent': xo.Ref(Quadrupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_quadrupole_entry.h"'
    ]


class ThinSliceQuadrupoleExit(
    _ThinSliceEdgeBase, _ThinSliceMultipoleEdgeBase, BeamElement
):
    _face = 'exit'

    _xofields = {'_parent': xo.Ref(Quadrupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_quadrupole_exit.h"'
    ]


class ThinSliceSextupoleEntry(
    _ThinSliceEdgeBase, _ThinSliceMultipoleEdgeBase, BeamElement
):
    _face = 'entry'

    _xofields = {'_parent': xo.Ref(Sextupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_sextupole_entry.h"'
    ]


class ThinSliceSextupoleExit(
    _ThinSliceEdgeBase, _ThinSliceMultipoleEdgeBase, BeamElement
):
    _face = 'exit'

    _xofields = {'_parent': xo.Ref(Sextupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_sextupole_exit.h"'
    ]


class ThinSliceOctupoleEntry(
    _ThinSliceEdgeBase, _ThinSliceMultipoleEdgeBase, BeamElement
):
    _face = 'entry'

    _xofields = {'_parent': xo.Ref(Octupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_octupole_entry.h"'
    ]


class ThinSliceOctupoleExit(
    _ThinSliceEdgeBase, _ThinSliceMultipoleEdgeBase, BeamElement
):
    _face = 'exit'

    _xofields = {'_parent': xo.Ref(Octupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_octupole_exit.h"'
    ]

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


class ThinSliceRBendEntry(_ThinSliceEdgeBase, _ThinSliceDipoleEdgeBase, BeamElement):
    _face = 'entry'

    _xofields = {'_parent': xo.Ref(RBend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_rbend_entry.h"'
    ]

    def _face_angle_offset(self):
        return self._parent.angle / 2

    def get_equivalent_element(self):

        _raise_if_parent_has_transverse_rotation(self._parent)

        if self._parent.rbend_model == "straight-body":
            return self # No replacement possible (not yet supported), element
                        # left where it is

        return _ThinSliceDipoleEdgeBase.get_equivalent_element(self)

    def _survey_start_end_elem(
            self,
            XYZ_ref_start,
            E_ref_start,
            XYZ_ref_end,
            E_ref_end,
    ):
        from ..survey.misalignment_survey import get_misaligned_element_survey

        (
            XYZ_elem_start,
            E_elem_start,
            XYZ_elem_end,
            E_elem_end,
        ) = get_misaligned_element_survey(
            self,
            XYZ_ref_start,
            E_ref_start,
            XYZ_ref_end,
            E_ref_end,
        )
        if self._parent.rbend_model != 'straight-body':
            return (
                XYZ_elem_start,
                E_elem_start,
                XYZ_elem_end,
                E_elem_end,
            )

        # The reference survey already applies the RBend entry map between
        # XYZ_ref_start and XYZ_ref_end. Therefore the generic physical exit
        # frame is the body-side frame; applying the parent entry transform
        # here would move it a second time.
        XYZ_body = XYZ_elem_end
        E_body = E_elem_end
        return XYZ_body, E_body, XYZ_body.copy(), E_body.copy()

    def track_frame(self, frame, backtrack=False):

        if self._parent.rbend_model == "straight-body":
            if backtrack:
                if abs(self._parent.angle) > 1e-10:  # avoid numerical issues
                    frame.translate_x(
                        self._parent._x0_in
                        * np.cos(self._parent.rot_s_rad))
                    frame.translate_y(
                        self._parent._x0_in
                        * np.sin(self._parent.rot_s_rad))
                frame.arc(
                    angle=-self._parent._angle_in,
                    tilt=self._parent.rot_s_rad,
                )
            else:
                frame.arc(
                    angle=self._parent._angle_in,
                    tilt=self._parent.rot_s_rad,
                )
                if abs(self._parent.angle) > 1e-10:  # avoid numerical issues
                    frame.translate_x(
                        -self._parent._x0_in
                        * np.cos(self._parent.rot_s_rad))
                    frame.translate_y(
                        -self._parent._x0_in
                        * np.sin(self._parent.rot_s_rad))

class ThinSliceRBendExit(_ThinSliceEdgeBase, _ThinSliceDipoleEdgeBase, BeamElement):
    _face = 'exit'

    _xofields = {'_parent': xo.Ref(RBend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_rbend_exit.h"'
    ]

    def _face_angle_offset(self):
        return self._parent.angle / 2

    def get_equivalent_element(self):

        _raise_if_parent_has_transverse_rotation(self._parent)

        if self._parent.rbend_model == "straight-body":
            return self # No replacement possible (not yet supported), element
                        # left where it is

        return _ThinSliceDipoleEdgeBase.get_equivalent_element(self)

    def _survey_start_end_elem(
            self,
            XYZ_ref_start,
            E_ref_start,
            XYZ_ref_end,
            E_ref_end,
    ):
        from ..survey.misalignment_survey import get_misaligned_element_survey

        (
            XYZ_elem_start,
            E_elem_start,
            XYZ_elem_end,
            E_elem_end,
        ) = get_misaligned_element_survey(
            self,
            XYZ_ref_start,
            E_ref_start,
            XYZ_ref_end,
            E_ref_end,
        )
        if self._parent.rbend_model != 'straight-body':
            return (
                XYZ_elem_start,
                E_elem_start,
                XYZ_elem_end,
                E_elem_end,
            )

        # The generic physical entrance frame is already on the body side of
        # the exit map. Collapse the zero-length edge slice onto that frame.
        XYZ_body = XYZ_elem_start
        E_body = E_elem_start
        return XYZ_body.copy(), E_body.copy(), XYZ_body, E_body

    def track_frame(self, frame, backtrack=False):

        if self._parent.rbend_model == "straight-body":
            if backtrack:
                frame.arc(
                    angle=-self._parent._angle_out,
                    tilt=self._parent.rot_s_rad,
                )
                if abs(self._parent.angle) > 1e-10:  # avoid numerical issues
                    frame.translate_x(
                        -self._parent._x0_out
                        * np.cos(self._parent.rot_s_rad))
                    frame.translate_y(
                        -self._parent._x0_out
                        * np.sin(self._parent.rot_s_rad))
            else:
                if abs(self._parent.angle) > 1e-10:  # avoid numerical issues
                    frame.translate_x(
                        self._parent._x0_out
                        * np.cos(self._parent.rot_s_rad))
                    frame.translate_y(
                        self._parent._x0_out
                        * np.sin(self._parent.rot_s_rad))
                frame.arc(
                    angle=self._parent._angle_out,
                    tilt=self._parent.rot_s_rad,
                )
