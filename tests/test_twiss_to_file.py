import xtrack as xt
import xobjects as xo
import numpy as np
import pytest
import pathlib

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

@pytest.mark.parametrize("check_type", ['csv', 'hdf5', 'tfs', 'json'])
def test_twiss_table_file(check_type, tmp_path):

    env = xt.load(test_data_folder / 'sps_thick/sps.seq')
    env.vars.load(test_data_folder / 'sps_thick/lhc_q20.str')
    line = env.sps
    line.set_particle_ref('p', energy0=26e9)
    tw = line.twiss4d()

    if check_type == 'csv':
        tw.to_csv(tmp_path / 'twiss_test.csv')
        tw_test = xt.load(tmp_path / 'twiss_test.csv')
    elif check_type == 'hdf5':
        tw.to_hdf5(tmp_path / 'twiss_test.h5')
        tw_test = xt.load(tmp_path / 'twiss_test.h5')
    elif check_type == 'tfs':
        tw.to_tfs(tmp_path / 'twiss_test.tfs')
        tw_test = xt.load(tmp_path / 'twiss_test.tfs')
    elif check_type == 'json':
        tw.to_json(tmp_path / 'twiss_test.json')
        tw_test = xt.load(tmp_path / 'twiss_test.json')
    else:
        raise ValueError(f'check_type {check_type} not supported')

    assert isinstance(tw_test, xt.TwissTable)

    assert np.all(np.array(tw_test._col_names) == np.array(tw._col_names))
    if check_type == 'tfs':
        assert set(tw_test.keys()) - set(tw.keys()) == {
            '__class__', 'attrs_serialization', 'column_serialization', 'xtrack_version'
        }
    elif check_type == 'json':
        assert set(tw_test.keys()) - set(tw.keys()) == {'xtrack_version'}
    else:
        assert set(tw_test.keys()) - set(tw.keys()) == {'__class__', 'xtrack_version'}
    assert set(tw.keys()) - set(tw_test.keys()) == {'_action'}

    for kk in tw._data:
        if kk == '_action':
            continue
        if kk in ['particle_on_co', 'steps_r_matrix', 'line_config']:
            continue # To be checked separately
        if tw[kk] is None:
            assert tw_test[kk] is None
            continue
        if isinstance(tw[kk], np.ndarray) and isinstance(tw[kk][0], str):
            assert np.all(tw[kk] == tw_test[kk])
            continue
        elif isinstance(tw[kk], str):
            assert tw[kk] == tw_test[kk]
            continue
        xo.assert_allclose(tw[kk], tw_test[kk], rtol=1e-7, atol=1e-15)

    # Check particle_on_co
    assert isinstance(tw.particle_on_co, xt.Particles)
    assert isinstance(tw_test.particle_on_co, xt.Particles)
    dct_ref = line.particle_ref.to_dict()
    dct_test = tw_test.particle_on_co.to_dict()
    for kk in dct_ref:
        if isinstance(dct_ref[kk], np.ndarray):
            xo.assert_allclose(dct_ref[kk], dct_test[kk], rtol=1e-10, atol=1e-15)
        else:
            assert dct_ref[kk] == dct_test[kk]

    # Check steps_r_matrix
    assert isinstance(tw.steps_r_matrix, dict)
    assert isinstance(tw_test.steps_r_matrix, dict)
    assert set(tw.steps_r_matrix.keys()) == set(tw_test.steps_r_matrix.keys())
    for kk in tw.steps_r_matrix:
        rmat_ref = tw.steps_r_matrix[kk]
        rmat_test = tw_test.steps_r_matrix[kk]
        xo.assert_allclose(rmat_ref, rmat_test, rtol=1e-10, atol=1e-15)

    # Check line_config
    assert isinstance(tw.line_config, dict)
    assert isinstance(tw_test.line_config, dict)
    assert set(tw.line_config.keys()) == set(tw_test.line_config.keys())
    for kk in tw.line_config:
        assert tw.line_config[kk] == tw_test.line_config[kk]