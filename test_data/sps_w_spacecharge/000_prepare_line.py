import json

import numpy as np

from cpymad.madx import Madx

import xpart as xp
import xobjects as xo
import xtrack as xt
import xfields as xf

import sc_tools as bt

seq_name = 'sps'
bunch_intensity = 1e11/3
sigma_z = 22.5e-2/3 # Short bunch to avoid probing bucket non-linearity
                    # to compare against frozen
neps_x=2.5e-6
neps_y=2.5e-6

mad = Madx()
mad.call('sps_thin.seq')
mad.use(seq_name)

line_without_spacecharge = xt.Line.from_madx_sequence(
                                            mad.sequence[seq_name],
                                            install_apertures=True)
# enable RF
V_RF = 3e6
i_cavity = line_without_spacecharge.element_names.index('acta.31637')
line_without_spacecharge.elements[i_cavity].voltage = V_RF
line_without_spacecharge.elements[i_cavity].lag = 180.

# enable RF in MAD-X
i_cav_madx = mad.sequence[seq_name].element_names().index('acta.31637')
mad.sequence[seq_name].elements[i_cav_madx].volt = V_RF/1e6
mad.sequence[seq_name].elements[i_cav_madx].lag = 0.5

mad.use(sequence=seq_name)
twiss_table = mad.twiss(rmatrix=True)

mad_beam =  mad.sequence[seq_name].beam
assert mad_beam.deltap == 0, "Not implemented."

particle_on_madx_co = xp.Particles(
    p0c = mad_beam.pc*1e9,
    q0 = mad_beam.charge,
    mass0 = mad_beam.mass*1e9,
    s = 0,
    x = twiss_table.x[0],
    px = twiss_table.px[0],
    y = twiss_table.y[0],
    py = twiss_table.py[0],
    tau = twiss_table.t[0],
    ptau = twiss_table.pt[0],
)

RR_madx = np.zeros([6,6])

for ii in range(6):
    for jj in range(6):
        RR_madx[ii, jj] = getattr(twiss_table, f're{ii+1}{jj+1}')[0]

optics_and_co_at_start_ring_from_madx = {
        'betx': twiss_table.betx[0],
        'bety': twiss_table.bety[0],
        'alfx': twiss_table.alfx[0],
        'alfy': twiss_table.alfy[0],
        'dx': twiss_table.dx[0],
        'dy': twiss_table.dy[0],
        'dpx': twiss_table.dpx[0],
        'dpy': twiss_table.dpy[0],
        'RR_madx': RR_madx,
        'particle_on_madx_co': particle_on_madx_co.to_dict()
        }

with open('optics_and_co_at_start_ring.json', 'w') as fid:
    json.dump(optics_and_co_at_start_ring_from_madx, fid, cls=xo.JEncoder)

part_on_co = xp.Particles.from_dict(
        optics_and_co_at_start_ring_from_madx['particle_on_madx_co'])

# A test particle
part = part_on_co.copy()
part.x += 2e-3
part.y += 3e-3
part.zeta += 20e-2
with open('line_no_spacecharge_and_particle.json', 'w') as fid:
    json.dump({
        'line': line_without_spacecharge.to_dict(),
        'particle': part.to_dict()},
        fid, cls=xo.JEncoder)

# Start space-charge configuration

# Make a matched bunch just to get the matched momentum spread
bunch = xp.generate_matched_gaussian_bunch(
         num_particles=int(2e6), total_intensity_particles=bunch_intensity,
         nemitt_x=neps_x, nemitt_y=neps_y, sigma_z=sigma_z,
         particle_ref=part_on_co, tracker=xt.Tracker(line=line_without_spacecharge))
delta_rms = np.std(bunch.delta)


sc_locations, sc_lengths = bt.determine_sc_locations(
    line=line_without_spacecharge,
    n_SCkicks = 540,
    length_fuzzy=0)

# Install spacecharge place holders
sc_names = ['sc%d' % number for number in range(len(sc_locations))]
bt.install_sc_placeholders(
    mad, 'sps', sc_names, sc_locations, mode='Bunched')

# Generate line with spacecharge
line_with_spacecharge = xt.Line.from_madx_sequence(
                                       mad.sequence['sps'],
                                       install_apertures=True)

# Get spacecharge names and twiss info from optics
mad_sc_names, sc_twdata = bt.get_spacecharge_names_twdata(
    mad, 'sps', mode='Bunched')

# Setup spacecharge
sc_elements, sc_names = line_with_spacecharge.get_elements_of_type(
       xf.SpaceChargeBiGaussian
    )
bt.setup_spacecharge_bunched_in_line(
        sc_elements=sc_elements,
        sc_lengths=sc_lengths,
        sc_twdata=sc_twdata,
        betagamma=part.beta0[0]*part.gamma0[0],
        number_of_particles=bunch_intensity,
        delta_rms=delta_rms,
        neps_x=neps_x,
        neps_y=neps_y,
        bunchlength_rms=sigma_z
    )

# enable RF
i_cavity = line_with_spacecharge.element_names.index('acta.31637')
line_with_spacecharge.elements[i_cavity].voltage = 3e6
line_with_spacecharge.elements[i_cavity].lag = 180.

with open('line_with_spacecharge_and_particle.json', 'w') as fid:
    json.dump({
        'line': line_with_spacecharge.to_dict(),
        'particle': part.to_dict()},
        fid, cls=xo.JEncoder)

