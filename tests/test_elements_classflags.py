import xtrack as xt

checks = [
    (xt.Drift,                   dict(isthick=True , allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.Multipole,               dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.Marker,                  dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.ReferenceEnergyIncrease, dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.Cavity,                  dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.XYShift,                 dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.ZetaShift,               dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.Elens,                   dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.Wire,                    dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.SRotation,               dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.YRotation,               dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.Solenoid,                dict(isthick=True , allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.RFMultipole,             dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.DipoleEdge,              dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.SimpleThinBend,          dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.SimpleThinQuadrupole,    dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.LineSegmentMap,          dict(isthick=True , allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.NonLinearLens,           dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.LimitEllipse,            dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.LimitRectEllipse,        dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.LimitRect,               dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.LimitRacetrack,          dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.LimitPolygon,            dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.DriftSlice,              dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=False, _inherit_strengths=False)),
    (xt.DriftSliceBend,          dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=False, _inherit_strengths=False)),
    (xt.DriftSliceOctupole,      dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=False, _inherit_strengths=False)),
    (xt.DriftSliceQuadrupole,    dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=False, _inherit_strengths=False)),
    (xt.DriftSliceSextupole,     dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=False, _inherit_strengths=False)),
    (xt.ThickSliceBend,          dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=True )),
    (xt.ThickSliceOctupole,      dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=True )),
    (xt.ThickSliceQuadrupole,    dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=True )),
    (xt.ThickSliceSextupole,     dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=True )),
    (xt.ThinSliceBend,           dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=True )),
    (xt.ThinSliceBendEntry,      dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=False)),
    (xt.ThinSliceBendExit,       dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=False)),
    (xt.ThinSliceOctupole,       dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=True )),
    (xt.ThinSliceQuadrupole,     dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=True )),
    (xt.ThinSliceSextupole,      dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=True )),
]

def test_elements_classflags():
    for cls, kwargs in checks:
        for kk, vv in kwargs.items():
            if vv is not None:
                assert getattr(cls, kk) == kwargs[kk], (
                    f'{cls.__name__}.{kk} is {getattr(cls, kk)} instead of {kwargs[kk]}')
            else:
                assert not hasattr(cls, kk), (
                    f'{cls.__name__}.{kk} should not be defined')