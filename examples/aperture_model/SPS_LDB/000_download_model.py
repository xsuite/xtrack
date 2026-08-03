import cernlayoutdb as layout
from pathlib import Path
import xtrack as xt


aperture_file = Path(__file__).parent / 'SPS.pickle'
if not aperture_file.exists():
    sps_ldb = layout.Machine.from_ldb("SPS", "LS3")
    sps_ldb.to_pickle(str(aperture_file))

lattice_file = Path(__file__).parent / 'sps.json'
if not lattice_file.exists():
    sps = xt.load('https://acc-models.web.cern.ch/acc-models/sps/2026/xsuite/sps.json')
    sps.vars.load('https://acc-models.web.cern.ch/acc-models/sps/2026/strengths/lhc_q20.str')
    sps.set_particle_ref('proton', p0c=26e9)
    sps.to_json('sps.json')

sps = xt.load(lattice_file)
