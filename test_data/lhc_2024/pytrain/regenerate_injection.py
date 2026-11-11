"""Regenerate ``pytrain_injection.json``: pytrain (TRAIN) reference for the
LHC injection multi-bunch beam-beam scenario (BBLR only, 45 LR/side at
IP1/2/5/8, 450 GeV, 1.8e11 p/bunch, 1.5 um, full 2460-bunch scheme, 3 solver
iterations) -- the same scenario as the xsuite example
``xtrack/examples/lhc_multibunch_bb/001_multibunch_sectormaps_injection.py``
and the test ``test_lhc_multibunch_train.py``.

Requires pytrain + cpymad (NOT installable in the xsuite environment), e.g.:
  cd /opt/mihostet/python/pytrain && venv39/bin/python <here>/regenerate_injection.py
"""
import os, json, time
import numpy as np
from cpymad import madx as cpymadx
from pytrain.cpymad import (cpymad_lhc_makethin, cpymad_lhc_install_bb_markers,
                            cpymad_lhc_cycle, cpymad_generate_train_maps)
from pytrain.machine import FillingScheme
from pytrain.solver import solve_train
from pytrain.twiss import oneturn_map, closed_orbit, tune

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)                     # test_data/lhc_2024
INT, EMIT, N_ITER = 1.8e11, 1.5e-6, 3
t0 = time.time()

madx = cpymadx.Madx(stdout=False)
madx.options.echo = madx.options.info = madx.options.warn = False
madx.call(os.path.join(DATA, 'lhc.seq'))
madx.call(os.path.join(DATA, 'injection_optics.madx'))
madx.input('beam,sequence=lhcb1,particle=proton,energy=NRJ;'
           'beam,sequence=lhcb2,particle=proton,energy=NRJ,bv=-1;')
cpymad_lhc_makethin(madx, slicefactor=8)
cpymad_lhc_install_bb_markers(madx, nparasitic=45)
cpymad_lhc_cycle(madx, start='IP3')
machine, tw1, tw2, maps1, maps2 = cpymad_generate_train_maps(madx,
                                                             num_slots=3564)
z1 = closed_orbit(oneturn_map(maps1)); qx0_1, qy0_1 = tune(oneturn_map(maps1), z1)
z2 = closed_orbit(oneturn_map(maps2)); qx0_2, qy0_2 = tune(oneturn_map(maps2), z2)
print(f'model ready {time.time()-t0:.0f}s  bare thin tunes '
      f'B1 {qx0_1:.5f}/{qy0_1:.5f}  B2 {qx0_2:.5f}/{qy0_2:.5f}', flush=True)

sc = json.load(open(os.path.join(
    DATA, '25ns_2460b_2448_2092_2239_144bpi_20inj.json')))
sb1 = np.array(sc['schemebeam1'], float)
sb2 = np.array(sc['schemebeam2'], float)
fs = FillingScheme(sb1*INT, sb2*INT, sb1*EMIT, sb1*EMIT, sb2*EMIT, sb2*EMIT)
res = solve_train(machine, fs, tw1, maps1, tw2, maps2,
                  tolerance=1e-9, max_iter=N_ITER)


def beam_dict(qx, qy, qx0, qy0, cox, coy):
    m = np.isfinite(qx)
    slots = np.where(m)[0]
    x, y = cox[m], coy[m]
    return {'slots': [int(v) for v in slots],
            'qx': list(qx[m]), 'qy': list(qy[m]),
            'dqx': list(qx[m] - qx0), 'dqy': list(qy[m] - qy0),
            'x': list(x), 'y': list(y),
            'dx': list(x - x.mean()), 'dy': list(y - y.mean())}


qx1, qy1 = res.bunch_tunes_b1(); qx2, qy2 = res.bunch_tunes_b2()
cox1, coy1 = res.bunch_positions_b1('MKIP1')
cox2, coy2 = res.bunch_positions_b2('MKIP1')
out = {
    'scenario': 'injection',
    'description': (
        'pytrain (thin sf8) reference, LHC injection 450 GeV, BBLR only '
        '(45 LR/side at IP1/2/5/8), 1.8e11 p/bunch, 1.5 um, full 2460-bunch '
        'scheme, 3 solver iterations. Orbits x/y at MKIP1 (physical frame), '
        'dx/dy = deviation from bunch average; qx/qy fractional tunes, '
        'dqx/dqy = shift vs bare thin lattice.'),
    'solver_iterations': N_ITER,
    'generated_by': 'test_data/lhc_2024/pytrain/regenerate_injection.py',
    'b1': beam_dict(qx1, qy1, qx0_1, qy0_1, cox1, coy1),
    'b2': beam_dict(qx2, qy2, qx0_2, qy0_2, cox2, coy2),
}
with open(os.path.join(HERE, 'pytrain_injection.json'), 'w') as fid:
    json.dump(out, fid)
print(f'DONE {time.time()-t0:.0f}s -> pytrain_injection.json', flush=True)
