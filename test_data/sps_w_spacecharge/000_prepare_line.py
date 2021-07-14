import json

import numpy as np

from cpymad.madx import Madx

import xline as xl
import xpart as xp
import pymask as pm

seq_name = 'sps'
bunch_intensity = 1e11
sigma_z = 22.5e-2
neps_x=2.5e-6
neps_y=2.5e-6

mad = Madx()
mad.call('sps_thin.seq')
mad.use(seq_name)

line_without_spacecharge = xl.Line.from_madx_sequence(
                                            mad.sequence[seq_name],
                                            install_apertures=True)
# enable RF in xline
V_RF = 3e6
i_cavity = line_without_spacecharge.element_names.index('acta.31637')
line_without_spacecharge.elements[i_cavity].voltage = V_RF
line_without_spacecharge.elements[i_cavity].lag = 180.

# enable RF in MAD-X
i_cav_madx = mad.sequence[seq_name].element_names().index('acta.31637')
mad.sequence[seq_name].elements[i_cav_madx].volt = V_RF/1e6
mad.sequence[seq_name].elements[i_cav_madx].lag = 0.5

# Optics and orbit at start ring
optics_and_orbit_at_start_ring = pm.get_optics_and_orbit_at_start_ring(
                                                  mad, seq_name=seq_name)
with open('optics_and_co_at_start_ring.json', 'w') as fid:
    json.dump(optics_and_orbit_at_start_ring, fid, cls=pm.JEncoder)

part_on_co = xp.Particles.from_dict(
        optics_and_orbit_at_start_ring['particle_on_madx_co'])
RR = np.array(optics_and_orbit_at_start_ring['RR_madx'])

# A test particle
part = part_on_co.copy()
part.x += 2e-3
part.y += 3e-3
part.zeta += 20e-2
with open('line_no_spacecharge_and_particle.json', 'w') as fid:
    json.dump({
        'line': line_without_spacecharge.to_dict(keepextra=True),
        'particle': part.to_dict()},
        fid, cls=pm.JEncoder)

# Start space-charge configuration

# Make a matched bunch just to get the matched momentum spread
bunch = xp.generate_matched_gaussian_bunch(
         num_particles=int(2e6), total_intensity_particles=bunch_intensity,
         nemitt_x=neps_x, nemitt_y=neps_y, sigma_z=sigma_z,
         particle_on_co=part_on_co, R_matrix=RR,
         circumference=mad.sequence[seq_name].beam.circ,
         alpha_momentum_compaction=mad.table.summ.alfa,
         rf_harmonic=4620, rf_voltage=V_RF, rf_phase=0)
delta_rms = np.std(bunch.delta)


import xline.be_beamfields.tools as bt
sc_locations, sc_lengths = bt.determine_sc_locations(
    line=line_without_spacecharge,
    n_SCkicks = 540,
    length_fuzzy=0)

# Install spacecharge place holders
sc_names = ['sc%d' % number for number in range(len(sc_locations))]
bt.install_sc_placeholders(
    mad, 'sps', sc_names, sc_locations, mode='Bunched')

# Generate line with spacecharge
line_with_spacecharge = xl.Line.from_madx_sequence(
                                       mad.sequence['sps'],
                                       install_apertures=True)

# Get spacecharge names and twiss info from optics
mad_sc_names, sc_twdata = bt.get_spacecharge_names_twdata(
    mad, 'sps', mode='Bunched')

# Setup spacecharge
sc_elements, sc_names = line_with_spacecharge.get_elements_of_type(
        xl.elements.SCQGaussProfile
    )
bt.setup_spacecharge_bunched_in_line(
        sc_elements=sc_elements,
        sc_lengths=sc_lengths,
        sc_twdata=sc_twdata,
        betagamma=part.beta0*part.gamma0,
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
        'line': line_with_spacecharge.to_dict(keepextra=True),
        'particle': part.to_dict()},
        fid, cls=pm.JEncoder)

