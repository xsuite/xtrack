import xtrack as xt

checks = [
    (xt.Drift,                   dict(_isthick=True, allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.Multipole,               dict(allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.Marker,                  dict(_isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.ReferenceEnergyIncrease, dict(_isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.Cavity,                  dict(_isthick=True, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.XYShift,                 dict(_isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.ZetaShift,               dict(_isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.Elens,                   dict(_isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.Wire,                    dict(_isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.SRotation,               dict(_isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.YRotation,               dict(_isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.Solenoid,                dict(_isthick=True , allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.UniformSolenoid,         dict(_isthick=True , allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.VariableSolenoid,        dict(_isthick=True , allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.RFMultipole,             dict(_isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.DipoleEdge,              dict(_isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.SimpleThinBend,          dict(_isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.SimpleThinQuadrupole,    dict(_isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.LineSegmentMap,          dict(_isthick=True , allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.NonLinearLens,           dict(_isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.LimitEllipse,            dict(_isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.LimitRectEllipse,        dict(_isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.LimitRect,               dict(_isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.LimitRacetrack,          dict(_isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.LimitPolygon,            dict(_isthick=False, allow_rot_and_shift=True , rot_and_shift_from_parent=None , _inherit_strengths=None)),
    (xt.DriftSlice,              dict(_isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=False, _inherit_strengths=False)),
    (xt.DriftSliceBend,          dict(_isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=False, _inherit_strengths=False)),
    (xt.DriftSliceOctupole,      dict(_isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=False, _inherit_strengths=False)),
    (xt.DriftSliceQuadrupole,    dict(_isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=False, _inherit_strengths=False)),
    (xt.DriftSliceSextupole,     dict(_isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=False, _inherit_strengths=False)),
    (xt.ThickSliceBend,          dict(_isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=True )),
    (xt.ThickSliceOctupole,      dict(_isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=True )),
    (xt.ThickSliceUniformSolenoid,dict(_isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=True )),
    (xt.ThickSliceQuadrupole,    dict(_isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=True )),
    (xt.ThickSliceSextupole,     dict(_isthick=True,  allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=True )),
    (xt.ThinSliceBend,           dict(_isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=True )),
    (xt.ThinSliceBendEntry,      dict(_isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=False)),
    (xt.ThinSliceBendExit,       dict(_isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=False)),
    (xt.ThinSliceOctupole,       dict(_isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=True )),
    (xt.ThinSliceQuadrupole,     dict(_isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=True )),
    (xt.ThinSliceSextupole,      dict(_isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=True )),
    (xt.ThinSliceUniformSolenoidEntry, dict(_isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=False)),
    (xt.ThinSliceUniformSolenoidExit,  dict(_isthick=False, allow_rot_and_shift=False, rot_and_shift_from_parent=True , _inherit_strengths=False)),
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
