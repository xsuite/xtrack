import pickle

import xtrack as xt
import xtrack.beam_elements as beam_elements
import xtrack.beam_elements.elements as elements


LEGACY_ELEMENTS_EXPORTS = {
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
    'ThinSliceNotNeededError',
}


def test_legacy_elements_exports_are_preserved():
    assert set(elements.__all__) == LEGACY_ELEMENTS_EXPORTS

    for name in LEGACY_ELEMENTS_EXPORTS:
        assert hasattr(elements, name)


def test_public_element_class_identity_is_preserved():
    for element_class in beam_elements.element_classes:
        name = element_class.__name__
        if hasattr(elements, name):
            assert getattr(elements, name) is element_class
        assert getattr(xt, name) is element_class

    assert len(beam_elements.element_classes) == len(
        set(beam_elements.element_classes)
    )


def test_pickle_from_legacy_elements_module_resolves():
    # Protocol 0 GLOBAL payload representing pickles produced when Drift lived
    # directly in xtrack.beam_elements.elements.
    legacy_drift_class_pickle = (
        b'cxtrack.beam_elements.elements\nDrift\n.'
    )
    assert pickle.loads(legacy_drift_class_pickle) is xt.Drift
