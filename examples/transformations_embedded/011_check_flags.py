import xtrack as xt

checks = [
    (xt.Drift,                   dict(isthick=True , allow_rot_and_shift=False, rot_and_shift_from_parent=None)),
    (xt.Multipole,               dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None)),
    (xt.Marker,                  dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None)),
    (xt.ReferenceEnergyIncrease, dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None)),
    (xt.Cavity,                  dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None)),
    (xt.XYShift,                 dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None)),
    (xt.ZetaShift,               dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None)),
    (xt.Elens,                   dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None)),
    (xt.Wire,                    dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None)),
    (xt.SRotation,               dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None)),
    (xt.YRotation,               dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None)),
    (xt.Solenoid,                dict(isthick=True , allow_rot_and_shift=True , rot_and_shift_from_parent=None)),
    (xt.RFMultipole,             dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None)),
    (xt.DipoleEdge,              dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None)),
    (xt.SimpleThinBend,          dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None)),
    (xt.SimpleThinQuadrupole,    dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None)),
    (xt.LineSegmentMap,          dict(isthick=True , allow_rot_and_shift=True , rot_and_shift_from_parent=None)),
    (xt.NonLinearLens,           dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None)),
    (xt.LimitEllipse,            dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None)),
    (xt.LimitRectEllipse,        dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None)),
    (xt.LimitRect,               dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None)),
    (xt.LimitRacetrack,          dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None)),
    (xt.LimitPolygon,            dict(isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None)),
    (xt.DriftSlice,              dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=False)),
    (xt.DriftSliceBend,          dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=False)),
    (xt.DriftSliceOctupole,      dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=False)),
    (xt.DriftSliceQuadrupole,    dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=False)),
    (xt.DriftSliceSextupole,     dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=False)),
    (xt.ThickSliceBend,          dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=True )),
    (xt.ThickSliceOctupole,      dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=True )),
    (xt.ThickSliceQuadrupole,    dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=True )),
    (xt.ThickSliceSextupole,     dict(isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=True )),
    (xt.ThinSliceBend,           dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True )),
    (xt.ThinSliceBendEntry,      dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True )),
    (xt.ThinSliceBendExit,       dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True )),
    (xt.ThinSliceOctupole,       dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True )),
    (xt.ThinSliceQuadrupole,     dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True )),
    (xt.ThinSliceSextupole,      dict(isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True )),
]

for cls, kwargs in checks:
    for kk, vv in kwargs.items():
        if vv is not None:
            assert getattr(cls, kk) == kwargs[kk], (
                f'{cls.__name__}.{kk} is {getattr(cls, kk)} instead of {kwargs[kk]}')
        else:
            assert not hasattr(cls, kk), (
                f'{cls.__name__}.{kk} should not be defined')