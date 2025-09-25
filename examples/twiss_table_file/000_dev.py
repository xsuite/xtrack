from cpymad.madx import Madx
import json
from pathlib import Path
import shlex
from typing import Dict, List, Optional

import xtrack as xt

BASE_DIR = Path(__file__).parent

mad = Madx()
mad.call(file="../../test_data/lhc_2024/lhc.seq")
mad.call(file="../../test_data/lhc_2024/injection_optics.madx")
mad.beam(particle="proton", energy=450)
mad.use(sequence="lhcb1")
mad.input('select, flag=twiss, column=name, keyword, s, betx, bety, mux, muy;')
mad.twiss(file=str(BASE_DIR / "twiss_lhcb1.tfs"))

lhc= xt.load('../../test_data/lhc_2024/lhc.seq')
lhc.vars.load('../../test_data/lhc_2024/injection_optics.madx')
lhc.set_particle_ref('proton', energy0=450e9)

tw1 = lhc.lhcb1.twiss4d()

col_out = ['name', 's', 'betx', 'bety', 'mux', 'muy']
scalar_out = ['qx', 'qy', 'dqx', 'dqy']

out = {}
out['scalar'] = {k: tw1[k] for k in scalar_out}
out['col'] = {k: tw1[k] for k in col_out}

xt.json.dump(out, str(BASE_DIR / 'twiss_lhcb1_xtrack.json'))