import itertools
import json
import os
from typing import Literal

import numpy as np
import xtrack as xt
import xobjects as xo

if not os.path.exists('json.lua'):
    os.system('curl -OL https://raw.githubusercontent.com/rxi/json.lua/master/json.lua')

MADNG_EXEC = '/opt/madng/madng'
MODE: Literal['save', 'discard'] = 'save'
XT_ELEMENTS = {
    'quadrupole': xt.Quadrupole,
    'sextupole': xt.Sextupole,
    'octupole': xt.Octupole,
}
WHICH_KN = {
    'quadrupole': 'k1',
    'sextupole': 'k2',
    'octupole': 'k3',
}

def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)

def coord_list(max_value):
    return list(np.linspace(-max_value, max_value, 7))

values_for_dims = [
    coord_list(1e-3),  # x
    coord_list(1e-5),  # px
    coord_list(1e-3),  # y
    coord_list(1e-5),  # py
    coord_list(1e-2),  # zeta
    coord_list(1e-3),  # delta
]
all_coords = np.array(list(itertools.product(*values_for_dims))).T

p = xt.Particles(
    x=all_coords[0],
    px=all_coords[1],
    y=all_coords[2],
    py=all_coords[3],
    zeta=all_coords[4],
    delta=all_coords[5],
    energy0=2e9,  # 2 GeV
    mass0=xt.ELECTRON_MASS_EV,
    q0=1,
    beta0=1,
)
p.to_json('initial_particles.json')

length = 0.01
knl = 1

for element in ('quadrupole', 'sextupole', 'octupole'):
    p_xt = {}

    for has_fringe in (True, False):
        _p = p.copy()
        line = xt.Line(
            elements=[
                XT_ELEMENTS[element](
                    length=length,
                    edge_entry_active=has_fringe,
                    edge_exit_active=has_fringe,
                    **{WHICH_KN[element]: knl / length},
                ),
            ],
            element_names=['q'],
        )
        line.build_tracker()
        line.track(_p)
        _p.sort()

        p_xt[has_fringe] = _p

    coords_xtrack_no_fringe = np.array([
        p_xt[0].x,
        p_xt[0].px,
        p_xt[0].y,
        p_xt[0].py,
        p_xt[0].zeta / p_xt[0].beta0,
        p_xt[0].ptau]
    )
    coords_xtrack_fringe = np.array([
        p_xt[1].x,
        p_xt[1].px,
        p_xt[1].y,
        p_xt[1].py,
        p_xt[1].zeta / p_xt[1].beta0,
        p_xt[1].ptau],
    )

    with open('config.json', 'w') as config_file:
        json.dump({
            'element': element,
            'param': WHICH_KN[element],
        }, config_file)

    if os.system(f'{MADNG_EXEC} fringe.lua'):
        raise RuntimeError("MAD-NG script failed, make sure to point "
                           "`MADNG_EXEC` to your madng executable path")

    p_fringe = load_json('track_fringe.json')
    p_no_fringe = load_json('track_no_fringe.json')

    coords_madng_fringe = np.array([p_fringe[coord] for coord in ('x', 'px', 'y', 'py', 't', 'pt')])
    coords_madng_no_fringe = np.array([p_no_fringe[coord] for coord in ('x', 'px', 'y', 'py', 't', 'pt')])

    print("Comparing mad-ng with and without fringes:")
    max_ng_errors = np.max(abs(coords_madng_fringe - coords_madng_no_fringe), axis=1)
    print(f"Max error: {', '.join(str(x) for x in max_ng_errors)}")

    print()

    print("Comparing mad-ng with no fringes and xtrack:")
    max_errors = np.max(abs(coords_madng_no_fringe - coords_xtrack_no_fringe), axis=1)
    print(f"Max error: {', '.join(str(x) for x in max_errors)}")

    avg_errors = np.mean(abs(coords_madng_no_fringe - coords_xtrack_no_fringe), axis=1)
    print(f"Avg error: {', '.join(str(x) for x in avg_errors)}")


    print()

    print("Comparing mad-ng and xtrack:")
    max_errors = np.max(abs(coords_madng_fringe - coords_xtrack_fringe), axis=1)
    print(f"Max error: {', '.join(str(x) for x in max_errors)}")

    avg_errors = np.mean(abs(coords_madng_fringe - coords_xtrack_fringe), axis=1)
    print(f"Avg error: {', '.join(str(x) for x in avg_errors)}")

    print()

    print("Particles after tracking:")
    print(f"Xtrack with fringes: {coords_xtrack_fringe}")
    print(f"Xtrack without fringes: {coords_xtrack_no_fringe}")
    print(f"MAD-NG with fringes: {coords_madng_fringe}")
    print(f"MAD-NG without fringes: {coords_madng_no_fringe}")

    print("Effect of the fringes")
    print(f"Xtrack with fringes - Xtrack without fringes: {coords_xtrack_fringe - coords_xtrack_no_fringe}")
    print(f"MAD-NG with fringes - MAD-NG without fringes: {coords_madng_fringe - coords_madng_no_fringe}")

    fringe_effect_madng = coords_madng_fringe - coords_madng_no_fringe
    if MODE == 'save':
        with open(f'{element}_fringe.json', 'w') as f:
            json.dump(fringe_effect_madng, f, indent=2, cls=xo.JEncoder)

        fringe_effect_madng = load_json(f'{element}_fringe.json')

    fringe_effect_xtrack = coords_xtrack_fringe - coords_xtrack_no_fringe

    xo.assert_allclose(fringe_effect_madng[0], fringe_effect_xtrack[0], atol=1e-16, rtol=1.3e-2)  # x
    xo.assert_allclose(fringe_effect_madng[1], fringe_effect_xtrack[1], atol=1e-16, rtol=1.3e-2)  # px
    xo.assert_allclose(fringe_effect_madng[2], fringe_effect_xtrack[2], atol=1e-16, rtol=1e-2)  # y
    xo.assert_allclose(fringe_effect_madng[3], fringe_effect_xtrack[3], atol=1e-16, rtol=1.1e-2)  # py
    xo.assert_allclose(fringe_effect_madng[4], fringe_effect_xtrack[4], atol=1e-15, rtol=1e-3)  # t
    xo.assert_allclose(fringe_effect_madng[5], fringe_effect_xtrack[5], atol=0, rtol=0)  # pt
