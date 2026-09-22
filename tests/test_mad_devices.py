import numpy as np
import pytest
from cpymad.madx import Madx

import xtrack as xt
from xtrack.mad_writer import MadType, element_to_mad_str


PASSIVE_TYPES = (
    'instrument', 'placeholder', 'monitor', 'hmonitor', 'vmonitor',
    'imonitor', 'collimator', 'rcollimator', 'ecollimator',
)


@pytest.mark.parametrize('loader', ['cpymad', 'native'])
@pytest.mark.parametrize('length', [0., 1.5])
def test_mad_passive_elements_are_devices(loader, length):
    types = PASSIVE_TYPES + (('elseparator',) if loader == 'cpymad'
                             else ('wire',))
    definitions = '\n'.join(
        f'dev_{kind}: {kind}, l:=device_length'
        + (', tilt:=device_tilt' if kind != 'imonitor' else '') + ';'
        for kind in types)
    placements = '\n'.join(
        f'dev_{kind}, at={3 * ii + 2};' for ii, kind in enumerate(types))
    source = f'''
        device_length={length}; device_tilt=0.12;
        {definitions}
        explicit_drift: drift, l=1;
        seq: sequence, l={3 * len(types) + 3};
        {placements}
        explicit_drift_start: marker, at={3 * len(types) + 0.5};
        explicit_drift, at={3 * len(types) + 1};
        endsequence;
    '''
    if loader == 'cpymad':
        with Madx(stdout=False) as mad:
            mad.input(source)
            mad.input('beam; use, sequence=seq;')
            line = xt.Line.from_madx_sequence(
                mad.sequence.seq, deferred_expressions=True, merge_drifts=True)
    else:
        line = xt.load(string=source, format='madx').seq

    for kind in types:
        device = line[f'dev_{kind}']
        assert isinstance(device, xt.Device)
        assert device.length == length
        assert device.rot_s_rad == (0 if kind == 'imonitor' else 0.12)

    assert isinstance(line['explicit_drift'], xt.Drift)
    assert any(isinstance(line[nn], xt.Drift)
               for nn in line.element_names if nn != 'explicit_drift')
    line['device_length'] = 0.7
    line['device_tilt'] = -0.2
    for kind in types:
        assert line[f'dev_{kind}'].length == 0.7
        assert line[f'dev_{kind}'].rot_s_rad == (
            0 if kind == 'imonitor' else -0.2)

    # Imported passive elements can now carry the offsets used by survey.
    line['dev_instrument'].shift_x = 2e-3
    line['dev_instrument'].shift_y = -1e-3
    survey = line.survey(include_element_frames=True)
    np.testing.assert_allclose(
        survey['XYZ_elem_start', 'dev_instrument']
        - survey['XYZ_ref_start', 'dev_instrument'],
        [2e-3, -1e-3, 0], atol=1e-14, rtol=0)


@pytest.mark.parametrize('deferred_expressions', [False, True])
@pytest.mark.parametrize('enable_align_errors', [False, True])
def test_cpymad_device_alignment_errors(deferred_expressions, enable_align_errors):
    with Madx(stdout=False) as mad:
        mad.input('''
            dev: instrument, l=1, tilt=0.1;
            seq: sequence, l=2;
            dev, at=1;
            endsequence;
            beam; use, sequence=seq;
            select, flag=error, pattern=dev;
            ealign, dx=0.002, dy=-0.001, dpsi=0.03;
        ''')
        line = xt.Line.from_madx_sequence(
            mad.sequence.seq, deferred_expressions=deferred_expressions,
            enable_align_errors=enable_align_errors)

    assert isinstance(line['dev'], xt.Device)
    assert line['dev'].rot_s_rad == pytest.approx(
        0.13 if enable_align_errors else 0.1)
    survey = line.survey(include_element_frames=True)
    expected = [2e-3, -1e-3, 0] if enable_align_errors else [0, 0, 0]
    np.testing.assert_allclose(
        survey['XYZ_elem_start', 'dev'] - survey['XYZ_ref_start', 'dev'],
        expected, atol=1e-14, rtol=0)


@pytest.mark.parametrize('loader', ['cpymad', 'native'])
def test_mad_device_aperture(loader):
    source = '''
        col: rcollimator, l=1, apertype=rectangle, aperture={0.02,0.01};
        seq: sequence, l=2;
        col, at=1;
        endsequence;
    '''
    if loader == 'cpymad':
        with Madx(stdout=False) as mad:
            mad.input(source)
            mad.input('beam; use, sequence=seq;')
            line = xt.Line.from_madx_sequence(mad.sequence.seq, install_apertures=True)
    else:
        line = xt.load(string=source, format='madx').seq
    assert isinstance(line['col'], xt.Device)
    aperture = line[line['col'].name_associated_aperture]
    assert isinstance(aperture, xt.LimitRect)
    assert aperture.max_x == 0.02
    assert aperture.max_y == 0.01


@pytest.mark.parametrize('length', [0., 1.5])
def test_device_madx_round_trip(length):
    env = xt.Environment()
    env['device_length'] = length
    env['device_tilt'] = 0.12
    line = env.new_line(components=[
        env.new('dev', 'Device', length='device_length', rot_s_rad='device_tilt'),
        env.new('dr', 'Drift', length=1),
    ])
    source = line.to_madx_sequence('seq')
    assert 'dev: instrument,' in source
    native = xt.load(string=source, format='madx').seq
    with Madx(stdout=False) as mad:
        mad.input(source)
        mad.input('beam; use, sequence=seq;')
        cpymad = xt.Line.from_madx_sequence(mad.sequence.seq, deferred_expressions=True)
    for imported in (native, cpymad):
        assert isinstance(imported['dev'], xt.Device)
        assert isinstance(imported['dr'], xt.Drift)
        assert imported['dev'].length == length
        assert imported['dev'].rot_s_rad == 0.12
        imported['device_length'] = 0.8
        imported['device_tilt'] = -0.2
        assert imported['dev'].length == 0.8
        assert imported['dev'].rot_s_rad == -0.2


@pytest.mark.parametrize('mad_type', [MadType.MADX, MadType.MADNG])
def test_device_writer_transforms(mad_type):
    line = xt.Line(elements={'dev': xt.Device(
        length=1.5, rot_s_rad=0.1, shift_x=0.002,
        shift_y=-0.001, shift_s=0.003, rot_s_rad_no_frame=0.04)})
    source = element_to_mad_str(
        'dev', 'dev', line, mad_type=mad_type, substituted_vars=[])
    assert source.startswith('instrument')
    for assignment in ('l = 1.5', 'tilt = 0.1', 'dx = 0.002',
                       'dy = -0.001', 'ds = 0.003'):
        assert assignment in source
    if mad_type == MadType.MADNG:
        assert 'misalign = MAD.typeid.deferred {' in source
        assert 'dpsi = 0.04' in source


def test_device_madng_export(sandbox_cwd):
    line = xt.Line(elements={'dev': xt.Device(length=1.5, shift_x=0.002)})
    line.particle_ref = xt.Particles(p0c=1e9)
    with line.to_madng() as mad:
        assert mad.seq.dev.l == 1.5
        assert mad.seq.dev.misalign.dx == 0.002


@pytest.mark.parametrize('mad_type', [MadType.MADX, MadType.MADNG])
def test_sliced_device_export(mad_type):
    line = xt.Line(elements={'dev': xt.Device(
        length=1.5, rot_s_rad=0.1, shift_x=0.002)})
    line.slice_thick_elements(slicing_strategies=[
        xt.Strategy(xt.Uniform(3, mode='thick'))])
    for name in ('dev..0', 'dev..1', 'dev..2'):
        source = element_to_mad_str(
            name, name, line, mad_type=mad_type, substituted_vars=[])
        assert source.startswith('instrument')
        assert 'l = 0.5' in source
        assert 'tilt = 0.1' in source
        assert 'dx = 0.002' in source
