import xtrack as xt
import xtrack.beam_elements as beam_elements


EXPECTED_BEAM_ELEMENT_EXPORTS = {
    'copy', 'List', 'warn', 'np', 'Number', 'factorial', 'xo', 'xt',
    'BeamElement', 'RandomUniformAccurate', 'RandomExponential',
    'RandomNormal', 'DEPRECATION_INFO_PREP_1_0', 'survey_advance_element',
    'RecordIndex', 'DEFAULT_MULTIPOLE_ORDER',
    'SynchrotronRadiationRecord', 'Spline4', 'SplineBoris',
    'ReferenceEnergyIncrease', 'ReferenceEnergyChange', 'Marker', 'Drift',
    'DriftExact', 'Cavity', 'CrabCavity', 'XYShift', 'Translation', 'Elens',
    'NonLinearLens', 'Wire', 'Rotation', 'SRotation', 'XRotation',
    'YRotation', 'ZetaShift', 'TimeDelay', 'Misalignment', 'Multipole',
    'SimpleThinQuadrupole', 'Bend', 'RBend', 'Sextupole', 'Octupole',
    'Quadrupole', 'UniformSolenoid', 'VariableSolenoid', 'TempRF',
    'Solenoid', 'Magnet', 'MagnetEdge', 'CombinedFunctionMagnet',
    'DipoleFringe', 'Wedge', 'SimpleThinBend', 'RFMultipole', 'DipoleEdge',
    'MultipoleEdge', 'LineSegmentMap', 'FirstOrderTaylorMap',
    'SecondOrderTaylorMap', 'ElectronCoolerRecord', 'ElectronCooler',
    'ThinSliceNotNeededError', 'UNLIMITED', 'LimitRect', 'LimitRacetrack',
    'LimitEllipse', 'LimitPolygon', 'LimitRectEllipse',
    'LongitudinalLimitRect',
}


def test_beam_element_exports_are_available():
    for name in EXPECTED_BEAM_ELEMENT_EXPORTS:
        assert hasattr(beam_elements, name)


def test_public_element_class_identity_is_preserved():
    for element_class in beam_elements.element_classes:
        name = element_class.__name__
        assert getattr(xt, name) is element_class

    assert len(beam_elements.element_classes) == len(
        set(beam_elements.element_classes)
    )


def test_aperture_imports_are_preserved():
    aperture_class_names = (
        'LimitRect', 'LimitRacetrack', 'LimitEllipse', 'LimitPolygon',
        'LimitRectEllipse', 'LongitudinalLimitRect',
    )
    for name in aperture_class_names:
        assert getattr(beam_elements, name) is getattr(xt, name)
