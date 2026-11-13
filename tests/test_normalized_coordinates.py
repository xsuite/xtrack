import numpy as np
import pytest

import xtrack as xt


@pytest.mark.parametrize('source', ['table', 'init'])
@pytest.mark.parametrize('emittances', [
    {},
    {'nemitt_x': 2.5e-6, 'nemitt_y': 1e-6, 'nemitt_zeta': 3e-4},
    {'nemitt_x': 2.5e-6},
    {'nemitt_y': 1e-6, 'nemitt_zeta': 3e-4},
])
def test_normalized_coordinate_emittances(source, emittances):
    co = xt.Particles(p0c=2e9, x=1e-3, y=-2e-3)
    diagonal = np.array([2., 0.5, 3., 1 / 3, 4., 0.25])
    W = np.diag(diagonal)
    coordinates = ['x', 'px', 'y', 'py', 'zeta', 'pzeta']
    if source == 'table':
        columns = {name: np.array(getattr(co, name)) for name in coordinates}
        columns.update(name=np.array(['start'], dtype=object), W_matrix=W[None])
        twiss = xt.TwissTable(
            {**columns, 'particle_on_co': co}, col_names=list(columns))
    else:
        twiss = xt.TwissInit(particle_on_co=co, W_matrix=W)

    particles = xt.Particles(
        p0c=2e9, x=[2e-3, 3e-3], px=[1e-4, -2e-4],
        y=[-3e-3, -1e-3], py=[3e-4, -4e-4],
        zeta=[0.01, -0.02], delta=[1e-3, -2e-3])
    result = twiss.get_normalized_coordinates(particles, **emittances)
    beta_gamma = co.beta0[0] * co.gamma0[0]

    for plane, names in zip(['x', 'y', 'zeta'],
                            [coordinates[:2], coordinates[2:4], coordinates[4:]]):
        key = f'nemitt_{plane}'
        emittance = emittances.get(key)
        assert getattr(result, key) == emittance
        assert key not in result._col_names
        assert result.to_dict()['attrs'][key] == emittance
        gemitt = 1 if emittance is None else emittance / beta_gamma
        for name in names:
            index = coordinates.index(name)
            expected = ((getattr(particles, name) - getattr(co, name))
                        / diagonal[index] / np.sqrt(gemitt))
            np.testing.assert_allclose(result[f'{name}_norm'], expected,
                                       rtol=1e-14, atol=1e-14)
