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
