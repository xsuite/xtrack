import xtrack as xt

checks = [
    (xt.Drift,                   dict(isthick=True , allow_rot_and_shift=False)),
    (xt.Multipole,               dict(isthick=False, allow_rot_and_shift=True )),
    (xt.Marker,                  dict(isthick=False, allow_rot_and_shift=False)),
    (xt.ReferenceEnergyIncrease, dict(isthick=False, allow_rot_and_shift=False)),
    (xt.Cavity,                  dict(isthick=False, allow_rot_and_shift=True)),
    (xt.XYShift,                 dict(isthick=False, allow_rot_and_shift=False)),
    (xt.ZetaShift,               dict(isthick=False, allow_rot_and_shift=False)),
    (xt.Elens,                   dict(isthick=False, allow_rot_and_shift=True)),
    (xt.Wire,                    dict(isthick=False, allow_rot_and_shift=True)),
    (xt.SRotation,               dict(isthick=False, allow_rot_and_shift=False)),
    (xt.YRotation,               dict(isthick=False, allow_rot_and_shift=False)),
    (xt.Solenoid,                dict(isthick=True , allow_rot_and_shift=True)),
    (xt.RFMultipole,             dict(isthick=False, allow_rot_and_shift=True)),
    (xt.DipoleEdge,              dict(isthick=False, allow_rot_and_shift=True)),
    (xt.SimpleThinBend,          dict(isthick=False, allow_rot_and_shift=False)),
    (xt.SimpleThinQuadrupole,    dict(isthick=False, allow_rot_and_shift=False)),
    (xt.LineSegmentMap,          dict(isthick=True , allow_rot_and_shift=True)),
    (xt.NonLinearLens,           dict(isthick=False, allow_rot_and_shift=True)),
    (xt.LimitEllipse,            dict(isthick=False, allow_rot_and_shift=True)),
    (xt.LimitRectEllipse,        dict(isthick=False, allow_rot_and_shift=True)),
    (xt.LimitRect,               dict(isthick=False, allow_rot_and_shift=True)),
    (xt.LimitRacetrack,          dict(isthick=False, allow_rot_and_shift=True)),
    (xt.LimitPolygon,            dict(isthick=False, allow_rot_and_shift=True)),
    (xt.DriftSlice,              dict(isthick=True,  allow_rot_and_shift=False)),
    (xt.DriftSliceBend,          dict(isthick=True,  allow_rot_and_shift=False)),
    (xt.DriftSliceOctupole,      dict(isthick=True,  allow_rot_and_shift=False)),
    (xt.DriftSliceQuadrupole,    dict(isthick=True,  allow_rot_and_shift=False)),
    (xt.DriftSliceSextupole,     dict(isthick=True,  allow_rot_and_shift=False)),
    (xt.ThickSliceBend,          dict(isthick=True,  allow_rot_and_shift=False)),
    (xt.ThickSliceOctupole,      dict(isthick=True,  allow_rot_and_shift=False)),
    (xt.ThickSliceQuadrupole,    dict(isthick=True,  allow_rot_and_shift=False)),
    (xt.ThickSliceSextupole,     dict(isthick=True,  allow_rot_and_shift=False)),
    (xt.ThinSliceBend,           dict(isthick=False, allow_rot_and_shift=False)),
    (xt.ThinSliceBendEntry,      dict(isthick=False, allow_rot_and_shift=False)),
    (xt.ThinSliceBendExit,       dict(isthick=False, allow_rot_and_shift=False)),
    (xt.ThinSliceOctupole,       dict(isthick=False, allow_rot_and_shift=False)),
    (xt.ThinSliceQuadrupole,     dict(isthick=False, allow_rot_and_shift=False)),
    (xt.ThinSliceSextupole,      dict(isthick=False, allow_rot_and_shift=False)),
]

for cls, kwargs in checks:
    for kk, vv in kwargs.items():
        if vv is not None:
            assert getattr(cls, kk) == kwargs[kk], (
                f'{cls.__name__}.{kk} is {getattr(cls, kk)} instead of {kwargs[kk]}')
        else:
            assert not hasattr(cls, kk), (
                f'{cls.__name__}.{kk} should not be defined')