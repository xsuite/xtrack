import cernlayoutdb as layout
from pathlib import Path
import requests


lhc = layout.Machine.from_ldb("LHC", "LS3")
lhc.to_pickle("LHC.pickle")

lattice_file = Path(__file__).parent / 'lhc.json'
if not lattice_file.exists():
    lattice_url = 'https://acc-models.web.cern.ch/acc-models/lhc/hl19/xsuite/lhc.json'
    request = requests.get(lattice_url, allow_redirects=True)
    lattice_file.write_bytes(request.content)

optics_file = Path(__file__).parent / 'opt_6000.madx'
if not optics_file.exists():
    optics_url = 'https://acc-models.web.cern.ch/acc-models/lhc/hl19/strengths/cycle_round_v3/opt_6000.madx'
    request = requests.get(optics_url, allow_redirects=True)
    optics_file.write_bytes(request.content)
