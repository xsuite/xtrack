"""Regenerate ``pytrain_collision.json`` (and the flattened xsuite optics
``../collision_optics_15cm_flat_2026.madx``): pytrain (TRAIN) reference for
the LHC collision multi-bunch beam-beam scenario -- 6.8 TeV, fully squeezed
R2025aRP 15 cm flat optics with end-of-levelling knobs (head-on at IP1/5,
levelling offsets at IP2/8), tunes/chroma matched to 62.316/60.322, Q'=10,
1.1e11 p/bunch, 2.3 um, full 2460-bunch scheme, 6 solver iterations. Same
scenario as the xsuite example
``xtrack/examples/lhc_multibunch_bb/002_multibunch_sectormaps_collisions.py``
and the test ``test_lhc_multibunch_train.py``.

The squeezed optics generate their knobs via MAD-X matching, which xsuite
cannot execute -- therefore this script also dumps the complete numeric
MAD-X global-variable state (after optics + knobs + matching) as the
flattened optics file used by the xsuite example. The acc-models-lhc optics
repository (public, tag pinned below) is fetched into a temporary directory
-- no local checkout is needed.

Requires pytrain + cpymad (NOT installable in the xsuite environment), e.g.:
  cd /opt/mihostet/python/pytrain && venv39/bin/python <here>/regenerate_collision.py
"""
import os, json, time
import tempfile
import urllib.request
import zipfile
import numpy as np
from cpymad import madx as cpymadx
from pytrain.cpymad import (cpymad_lhc_makethin, cpymad_lhc_install_bb_markers,
                            cpymad_lhc_cycle, cpymad_generate_train_maps,
                            cpymad_lhc_match_tune_chroma)
from pytrain.machine import FillingScheme
from pytrain.solver import solve_train
from pytrain.twiss import oneturn_map, closed_orbit, tune

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)                     # test_data/lhc_2024
ACC_MODELS_ZIP = ('https://gitlab.cern.ch/acc-models/acc-models-lhc/-/'
                  'archive/v2026.6/acc-models-lhc-v2026.6.zip')
INT, EMIT, N_ITER = 1.1e11, 2.3e-6, 6
t0 = time.time()

# Fetch the optics model into a temporary directory (as in the notebook
# examples); the archive extracts to acc-models-lhc-<tag>/, renamed to
# acc-models-lhc/ so the relative paths inside the optics files resolve.
modeldir = tempfile.TemporaryDirectory(prefix='acc-models-lhc-')
print(f'fetching {ACC_MODELS_ZIP} ...', flush=True)
zpath = os.path.join(modeldir.name, 'acc-models-lhc.zip')
urllib.request.urlretrieve(ACC_MODELS_ZIP, zpath)
with zipfile.ZipFile(zpath) as zf:
    zf.extractall(modeldir.name)
os.remove(zpath)
extracted, = [d for d in os.listdir(modeldir.name)
              if d.startswith('acc-models-lhc')]
os.rename(os.path.join(modeldir.name, extracted),
          os.path.join(modeldir.name, 'acc-models-lhc'))
print(f'model ready in {modeldir.name} ({time.time()-t0:.0f}s)', flush=True)

madx = cpymadx.Madx(stdout=False)
madx.chdir(modeldir.name)
madx.options.echo = madx.options.info = madx.options.warn = False
madx.call('acc-models-lhc/lhc.seq')
madx.input('''
beam, sequence=lhcb1, bv= 1, particle=proton, charge=1, mass=0.938272046,
  energy=6800, npart=1.2e11, kbunch=2400;
beam, sequence=lhcb2, bv=-1, particle=proton, charge=1, mass=0.938272046,
  energy=6800, npart=1.2e11, kbunch=2400;
''')
madx.call('acc-models-lhc/operation/optics/'
          'R2025aRP_A15cmC15cmA10mL200cm_Flat.madx')

# knobs (end-of-levelling scenario)
madx.globals.on_x1_v = 0
madx.globals.on_x5_h = 0
madx.globals.on_x1_h = 120
madx.globals.on_x5_v = -120
madx.globals.on_sep1_h = 0
madx.globals.on_sep5_v = 0
madx.globals.on_sep2h = -0.1
madx.globals.on_sep2v = 0
madx.globals.on_x2v = 200
madx.globals.on_sep8h = -0.02
madx.globals.on_sep8v = 0.01
madx.globals.on_x8h = 0
madx.globals.on_x8v = 200
madx.globals.on_a2 = 0
madx.globals.on_a8 = 0
madx.globals.on_disp = 1
madx.globals.on_alice = '7000/nrj'
madx.globals.on_lhcb = '7000/nrj'
madx.globals.on_sol_atlas = '7000/nrj'
madx.globals.on_sol_cms = '7000/nrj'
madx.globals.on_sol_alice = '7000/nrj'
madx.globals['KOF.B1'] = -12
madx.globals['KOD.B1'] = -12
madx.globals['KOF.B2'] = -12
madx.globals['KOD.B2'] = -12
print(f'optics + knobs {time.time()-t0:.0f}s', flush=True)

cpymad_lhc_makethin(madx, slicefactor=8)
cpymad_lhc_cycle(madx, start='IP3')
cpymad_lhc_install_bb_markers(madx, nparasitic=45)
cpymad_lhc_match_tune_chroma(madx, qx_b1=62.316, qy_b1=60.322,
                             qx_b2=62.316, qy_b2=60.322,
                             qpx_b1=10.0, qpy_b1=10.0,
                             qpx_b2=10.0, qpy_b2=10.0, tolerance=1e-4)
print(f'thin + markers + matched {time.time()-t0:.0f}s', flush=True)

# dump the full numeric global-variable state as a flattened optics file
knobs = {}
for name in madx.globals:
    if name == 'version':
        continue
    try:
        knobs[name] = float(madx.globals[name])
    except (TypeError, ValueError):
        pass
with open(os.path.join(DATA, 'collision_optics_15cm_flat_2026.madx'),
          'w') as fid:
    fid.write(
        "! Flattened LHC collision optics, 6.8 TeV, "
        "R2025aRP_A15cmC15cmA10mL200cm_Flat\n"
        "! (acc-models-lhc branch 2026) with end-of-levelling knobs and "
        "tunes/chroma\n! matched to 62.316/60.322, Q'=10. Generated by\n"
        "! test_data/lhc_2024/pytrain/regenerate_collision.py\n\n")
    for kk, vv in knobs.items():
        fid.write(f'{kk} := {vv!r} ;\n')
print(f'dumped {len(knobs)} globals to '
      f'collision_optics_15cm_flat_2026.madx', flush=True)

machine, tw1, tw2, maps1, maps2 = cpymad_generate_train_maps(madx,
                                                             num_slots=3564)
z1 = closed_orbit(oneturn_map(maps1)); qx0_1, qy0_1 = tune(oneturn_map(maps1), z1)
z2 = closed_orbit(oneturn_map(maps2)); qx0_2, qy0_2 = tune(oneturn_map(maps2), z2)
print(f'bare thin tunes B1 {qx0_1:.5f}/{qy0_1:.5f}  '
      f'B2 {qx0_2:.5f}/{qy0_2:.5f}', flush=True)

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
    'scenario': 'collision',
    'description': (
        'pytrain (thin sf8) reference, LHC collision 6.8 TeV R2025aRP 15 cm '
        'flat optics with end-of-levelling knobs (head-on IP1/5 + BBLR, '
        '45 LR/side at IP1/2/5/8), 1.1e11 p/bunch, 2.3 um, full 2460-bunch '
        'scheme, 6 solver iterations. Same column conventions as injection.'),
    'solver_iterations': N_ITER,
    'generated_by': 'test_data/lhc_2024/pytrain/regenerate_collision.py',
    'b1': beam_dict(qx1, qy1, qx0_1, qy0_1, cox1, coy1),
    'b2': beam_dict(qx2, qy2, qx0_2, qy0_2, cox2, coy2),
}
with open(os.path.join(HERE, 'pytrain_collision.json'), 'w') as fid:
    json.dump(out, fid)
print(f'DONE {time.time()-t0:.0f}s -> pytrain_collision.json', flush=True)
