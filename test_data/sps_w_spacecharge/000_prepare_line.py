import json

import numpy as np

from cpymad.madx import Madx

import xpart as xp
import xobjects as xo
import xtrack as xt
import xfields as xf


seq_name = 'sps'
bunch_intensity = 1e11/3
sigma_z = 22.5e-2/3 # Short bunch to avoid probing bucket non-linearity
                    # to compare against frozen
nemitt_x=2.5e-6
nemitt_y=2.5e-6

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

lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles=bunch_intensity,
        sigma_z=sigma_z,
        z0=0.,
        q_parameter=1.)

line_with_spacecharge = xf.install_spacecharge_frozen(
            line=line_without_spacecharge,
            particle_ref=part,
            longitudinal_profile=lprofile,
            nemitt_x=nemitt_x, nemitt_y=nemitt_y,
            sigma_z=sigma_z,
            num_spacecharge_interactions=540,
            tol_spacecharge_position=0)

with open('line_with_spacecharge_and_particle.json', 'w') as fid:
    json.dump({
        'line': line_with_spacecharge.to_dict(),
        'particle': part.to_dict()},
        fid, cls=xo.JEncoder)