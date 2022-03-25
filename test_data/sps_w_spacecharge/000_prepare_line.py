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

line = xt.Line.from_madx_sequence(
                                            mad.sequence[seq_name],
                                            install_apertures=True)
# enable RF
V_RF = 3e6
line['acta.31637'].voltage = V_RF
line['acta.31637'].lag = 180.

# A test particle
part = xp.Particles(gamma0=mad.sequence[seq_name].beam.gamma,
                    mass0=xp.PROTON_MASS_EV)
part.x += 2e-3
part.y += 3e-3
part.zeta += 20e-2
with open('line_no_spacecharge_and_particle.json', 'w') as fid:
    json.dump({
        'line': line.to_dict(),
        'particle': part.to_dict()},
        fid, cls=xo.JEncoder)

lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles=bunch_intensity,
        sigma_z=sigma_z,
        z0=0.,
        q_parameter=1.)

xf.install_spacecharge_frozen(
            line=line,
            particle_ref=part,
            longitudinal_profile=lprofile,
            nemitt_x=nemitt_x, nemitt_y=nemitt_y,
            sigma_z=sigma_z,
            num_spacecharge_interactions=540,
            tol_spacecharge_position=0)

with open('line_with_spacecharge_and_particle.json', 'w') as fid:
    json.dump({
        'line': line.to_dict(),
        'particle': part.to_dict()},
        fid, cls=xo.JEncoder)
