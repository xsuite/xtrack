import xtrack as xt

checks = [
    (xt.Drift,                   dict(isthick=True , )),
    (xt.Multipole,               dict(isthick=False, )),
    (xt.Marker,                  dict(isthick=False, )),
    (xt.ReferenceEnergyIncrease, dict(isthick=False, )),
    (xt.Cavity,                  dict(isthick=False, )),
    (xt.XYShift,                 dict(isthick=False, )),
    (xt.ZetaShift,               dict(isthick=False, )),
    (xt.Elens,                   dict(isthick=False, )),
    (xt.Wire,                    dict(isthick=False, )),
    (xt.SRotation,               dict(isthick=False, )),
    (xt.YRotation,               dict(isthick=False, )),
    (xt.Solenoid,                dict(isthick=True , )),
    (xt.RFMultipole,             dict(isthick=False, )),
    (xt.DipoleEdge,              dict(isthick=False, )),
    (xt.SimpleThinBend,          dict(isthick=False, )),
    (xt.SimpleThinQuadrupole,    dict(isthick=False, )),
    (xt.LineSegmentMap,          dict(isthick=True , )),
    (xt.NonLinearLens,           dict(isthick=False, )),
    (xt.LimitEllipse,            dict(isthick=False, )),
    (xt.LimitRectEllipse,        dict(isthick=False, )),
    (xt.LimitRect,               dict(isthick=False, )),
    (xt.LimitRacetrack,          dict(isthick=False, )),
    (xt.LimitPolygon,            dict(isthick=False, )),
    (xt.DriftSlice,              dict(isthick=True,  )),
    (xt.DriftSliceBend,          dict(isthick=True,  )),
    (xt.DriftSliceOctupole,      dict(isthick=True,  )),
    (xt.DriftSliceQuadrupole,    dict(isthick=True,  )),
    (xt.DriftSliceSextupole,     dict(isthick=True,  )),
    (xt.ThickSliceBend,          dict(isthick=True,  )),
    (xt.ThickSliceOctupole,      dict(isthick=True,  )),
    (xt.ThickSliceQuadrupole,    dict(isthick=True,  )),
    (xt.ThickSliceSextupole,     dict(isthick=True,  )),
    (xt.ThinSliceBend,           dict(isthick=False, )),
    (xt.ThinSliceBendEntry,      dict(isthick=False, )),
    (xt.ThinSliceBendExit,       dict(isthick=False, )),
    (xt.ThinSliceOctupole,       dict(isthick=False, )),
    (xt.ThinSliceQuadrupole,     dict(isthick=False, )),
    (xt.ThinSliceSextupole,      dict(isthick=False, )),
]

for cls, kwargs in checks:
    for kk, vv in kwargs.items():
        if vv is not None:
            assert getattr(cls, kk) == kwargs[kk], (
                f'{cls.__name__}.{kk} is {getattr(cls, kk)} instead of {kwargs[kk]}')