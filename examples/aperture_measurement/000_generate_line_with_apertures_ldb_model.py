import xtrack as xt
import numpy as np
import xobjects as xo

# Aperture markers with known issues
aper_blacklist = [
    'vtaf.51632.b_aper', 'vbrta.51633.a_aper', 'vbrta.51633.b_aper',
       'bgiha.51634.a_aper', 'bgiva.51674.a_aper']

# Load lattice and optics
env = xt.load(['../../test_data/sps_with_apertures/EYETS 2024-2025.seq',
               '../../test_data/sps_with_apertures/lhc_q20.str'])
line = env.sps
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, p0c=26e9)

tw0 = line.twiss4d()

# Load aperture markers in a dummy sequence
from cpymad.madx import Madx
mad = Madx()
mad.input('''
SPS : SEQUENCE, refer = centre,    L = 7000;
a: marker, at = 20;
endsequence;
''')
mad.call('../../test_data/sps_with_apertures/APERTURE_EYETS 2024-2025.seq')
mad.beam()
mad.use('SPS')
line_aper = xt.Line.from_madx_sequence(mad.sequence.SPS, install_apertures=True)

# Identify the aperture markers
tt_aper = line_aper.get_table().rows['.*_aper']

# Prepare insertions
insertions = []
for nn in tt_aper.name:
    if nn in aper_blacklist:
        continue
    env.elements[nn] = line_aper.get(nn).copy()
    insertions.append(env.place(nn, at=tt_aper['s', nn]))

# Shorten all pipes by 1 mm to avoid overlaps
for ins in insertions:
    if ins.name.endswith('.a_aper'):
        ins.at += 1e-3
    if ins.name.endswith('.b_aper'):
        ins.at -= 1e-3

# Insert the apertures into the line
line = env.sps
line.insert(insertions)

# Save
line.to_json('sps_with_apertures_ldb_model.json.gz')
