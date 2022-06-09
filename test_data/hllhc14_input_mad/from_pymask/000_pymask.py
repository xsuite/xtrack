# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import os
import sys
import json
import yaml
import numpy as np


#####################################################
# Read general configurations and setup envirnoment #
#####################################################

assert not(os.path.isfile('config.yaml') and os.path.isfile('config.py')), \
        "Please specify only a config file (yaml or py)"

try:
    with open('config.yaml','r') as fid:
        configuration = yaml.safe_load(fid)
except:
    from config import configuration

mode = configuration['mode']
tol_beta = configuration['tol_beta']
tol_sep = configuration['tol_sep']
flat_tol = configuration['tol_co_flatness']
links = configuration['links']
optics_file = configuration['optics_file']
check_betas_at_ips = configuration['check_betas_at_ips']
check_separations_at_ips = configuration['check_separations_at_ips']
save_intermediate_twiss = configuration['save_intermediate_twiss']
enable_lumi_control = configuration['enable_lumi_control']
enable_imperfections = configuration['enable_imperfections']
enable_crabs = configuration['enable_crabs']
match_q_dq_with_bb = configuration['match_q_dq_with_bb']
knob_settings = configuration['knob_settings']
knob_names = configuration['knob_names']


# Make links
for kk in links.keys():
    if os.path.exists(kk):
        os.remove(kk)
    os.symlink(os.path.abspath(links[kk]), kk)

# Create empty temp folder
os.system('rm -r temp')
os.system('mkdir temp')

# Execute customization script if present
os.system('bash customization.bash')

# Import pymask
sys.path.append('./modules')
import pymask as pm

# Import user-defined optics-specific tools
import optics_specific_tools as ost

######################################
# Check parameters and activate mode #
######################################

# Define configuration
(beam_to_configure, sequences_to_check, sequence_to_track, generate_b4_from_b2,
    track_from_b4_mad_instance, enable_bb_python, enable_bb_legacy,
    force_disable_check_separations_at_ips,
    ) = pm.get_pymask_configuration(mode)

if force_disable_check_separations_at_ips:
    check_separations_at_ips = False

if not(enable_crabs):
    knob_settings['par_crab1'] = 0.
    knob_settings['par_crab5'] = 0.

########################
# Build MAD-X instance #
########################

# Start mad
Madx = pm.Madxp
mad = Madx(command_log="mad_collider.log")

# Set verbose flag
mad.globals.par_verbose = int(configuration['verbose_mad_parts'])

# Build sequence (alse creates link to optics_toolkit and calls it)
ost.build_sequence(mad, beam=beam_to_configure)

# Set twiss formats for MAD-X parts (macro from opt. toolkit)
mad.input('exec, twiss_opt;')

# Apply optics
ost.apply_optics(mad, optics_file=optics_file)

# Attach beam to sequences
mad.globals.nrj = configuration['beam_energy_tot']
particle_type = 'proton'

if 'particle_mass' in configuration.keys():
    particle_mass = configuration['particle_mass']
    particle_type = 'ion'
else:
    particle_mass = mad.globals.pmass # proton mass

if 'particle_charge' in configuration.keys():
    particle_charge = configuration['particle_charge']
    particle_type = 'ion'
else:
    particle_charge = 1.

gamma_rel = (particle_charge*configuration['beam_energy_tot'])/particle_mass
for ss in mad.sequence.keys():
    # bv and bv_aux flags
    if ss == 'lhcb1':
        ss_beam_bv, ss_bv_aux = 1, 1
    elif ss == 'lhcb2':
        if int(beam_to_configure) == 4:
            ss_beam_bv, ss_bv_aux = 1, -1
        else:
            ss_beam_bv, ss_bv_aux = -1, 1

    mad.globals['bv_aux'] = ss_bv_aux
    mad.input(f'''
    beam, particle={particle_type},sequence={ss},
        energy={configuration['beam_energy_tot']*particle_charge},
        sigt={configuration['beam_sigt']},
        bv={ss_beam_bv},
        npart={configuration['beam_npart']},
        sige={configuration['beam_sige']},
        ex={configuration['beam_norm_emit_x'] * 1e-6 / gamma_rel},
        ey={configuration['beam_norm_emit_y'] * 1e-6 / gamma_rel},
        mass={particle_mass},
        charge={particle_charge};
    ''')


# Test machine before any change
twiss_dfs, other_data = ost.twiss_and_check(mad, sequences_to_check,
        tol_beta=tol_beta, tol_sep=tol_sep,
        twiss_fname='twiss_from_optics',
        save_twiss_files=save_intermediate_twiss,
        check_betas_at_ips=check_betas_at_ips,
        check_separations_at_ips=check_separations_at_ips)

# Set IP1-IP5 phase and store corresponding reference
mad.input("call, file='modules/submodule_01c_phase.madx';")

# Set optics-specific knobs
ost.set_optics_specific_knobs(mad, knob_settings, mode)

# Crossing-save and some reference measurements
mad.input('exec, crossing_save;')
mad.input("call, file='modules/submodule_01e_final.madx';")


#################################
# Check bahavior of orbit knobs #
#################################

# Check flat machine
mad.input('exec, crossing_disable;')
twiss_dfs, other_data = ost.twiss_and_check(mad, sequences_to_check,
        tol_beta=tol_beta, tol_sep=tol_sep,
        twiss_fname='twiss_no_crossing',
        save_twiss_files=save_intermediate_twiss,
        check_betas_at_ips=check_betas_at_ips, check_separations_at_ips=check_separations_at_ips)

# Check orbit flatness
for ss in twiss_dfs.keys():
    tt = twiss_dfs[ss]
    assert np.max(np.abs(tt.x)) < flat_tol
    assert np.max(np.abs(tt.y)) < flat_tol

# Check machine after crossing restore
mad.input('exec, crossing_restore;')
twiss_dfs, other_data = ost.twiss_and_check(mad, sequences_to_check,
        tol_beta=tol_beta, tol_sep=tol_sep,
        twiss_fname='twiss_with_crossing',
        save_twiss_files=save_intermediate_twiss,
        check_betas_at_ips=check_betas_at_ips, check_separations_at_ips=check_separations_at_ips)


#################################
# Set luminosity in IP2 and IP8 #
#################################

if len(sequences_to_check) == 2:
    print('Luminosities before leveling (crab cavities are not considered):')
    pm.print_luminosity(mad, twiss_dfs,
            configuration['nco_IP1'], configuration['nco_IP2'],
            configuration['nco_IP5'], configuration['nco_IP8'])
else:
    print('Warning: Luminosity computation requires two beams')


if not enable_lumi_control:
    print('Separations in IP2 and IP8 are left untouched')
elif enable_bb_legacy or mode=='b4_without_bb':
    mad.use(f'lhcb{beam_to_configure}')
    if mode=='b4_without_bb':
        print('Leveling not working in this mode!')
    else:
        if particle_type == 'ion': # the legacy macro for BB have been checked but not maintained
            raise ValueError
        # Luminosity levelling
        vars_for_legacy_level = ['lumi_ip8',
            'nco_IP1', 'nco_IP2', 'nco_IP5', 'nco_IP8']
        mad.set_variables_from_dict({
            'par_'+kk: configuration[kk] for kk in vars_for_legacy_level})
        mad.input("call, file='modules/module_02_lumilevel.madx';")
else:
    print('Start pythonic leveling:')
    ost.lumi_control(mad, twiss_dfs, configuration, knob_names)

# Force leveling
if 'force_leveling' in configuration.keys():
    force_leveling  = configuration['force_leveling']
    if force_leveling is not None:
        for kk in force_leveling.keys():
            mad.globals[kk] = force_leveling[kk]

# Re-save knobs (for the last time!)
mad.input('exec, crossing_save;')

# Check machine after leveling
mad.input('exec, crossing_restore;')
twiss_dfs, other_data = ost.twiss_and_check(mad, sequences_to_check,
        tol_beta=tol_beta, tol_sep=tol_sep,
        twiss_fname='twiss_after_leveling',
        save_twiss_files=save_intermediate_twiss,
        check_betas_at_ips=check_betas_at_ips,
        check_separations_at_ips=check_separations_at_ips)

if len(sequences_to_check) == 2:
    print('Luminosities after leveling (crab cavities are not considered):')
    pm.print_luminosity(mad, twiss_dfs,
            configuration['nco_IP1'], configuration['nco_IP2'],
            configuration['nco_IP5'], configuration['nco_IP8'])
else:
    print('Luminosity computation requires two beams')


#####################
# Force on_disp = 0 #
#####################

mad.globals.on_disp = 0.
# will be restored later


###################################
# Compute beam-beam configuration #
###################################

# Prepare bb dataframes
if enable_bb_python:
    bbconfig = configuration['beambeam_config']
    bb_dfs = pm.generate_bb_dataframes(mad,
        ip_names=['ip1', 'ip2', 'ip5', 'ip8'],
        harmonic_number=35640,
        numberOfLRPerIRSide=bbconfig['numberOfLRPerIRSide'],
        bunch_spacing_buckets=bbconfig['bunch_spacing_buckets'],
        numberOfHOSlices=bbconfig['numberOfHOSlices'],
        bunch_num_particles = bbconfig['bunch_num_particles'],
        bunch_particle_charge = bbconfig['bunch_particle_charge'],
        sigmaz_m=bbconfig['sigmaz_m'],
        z_crab_twiss=bbconfig['z_crab_twiss']*float(enable_crabs),
        remove_dummy_lenses=True)

    # Here the datafremes can be edited, e.g. to set bbb intensity

###################
# Generate beam 4 #
###################

if generate_b4_from_b2:
    mad_b4 = Madx(command_log="mad_b4.log")
    ost.build_sequence(mad_b4, beam=4)

    pm.configure_b4_from_b2(mad_b4, mad)

    twiss_dfs_b2, other_data_b2 = ost.twiss_and_check(mad,
            sequences_to_check=['lhcb2'],
            tol_beta=tol_beta, tol_sep=tol_sep,
            twiss_fname='twiss_b2_for_b4check',
            save_twiss_files=save_intermediate_twiss,
            check_betas_at_ips=check_betas_at_ips, check_separations_at_ips=False)

    twiss_dfs_b4, other_data_b4 = ost.twiss_and_check(mad_b4,
            sequences_to_check=['lhcb2'],
            tol_beta=tol_beta, tol_sep=tol_sep,
            twiss_fname='twiss_b4_for_b4check',
            save_twiss_files=save_intermediate_twiss,
            check_betas_at_ips=check_betas_at_ips, check_separations_at_ips=False)

# For B1, to be generalized for B4
if 'filling_scheme_json' in configuration['beambeam_config'].keys():
    assert 'b4' not in mode
    filling_scheme_json = configuration['beambeam_config']['filling_scheme_json']
    bunch_to_track = configuration['beambeam_config']['bunch_to_track']
    bb_schedule_to_track_b1 = ost.create_bb_shedule_to_track(
                              filling_scheme_json,bunch_to_track, beam=1)
    bb_dfs['b1']=ost.filter_bb_df(bb_dfs['b1'],bb_schedule_to_track_b1)

##################################################
# Select mad instance for tracking configuration #
##################################################

# We will be working exclusively on the sequence to track
# Select mad object
if track_from_b4_mad_instance:
    mad_track = mad_b4
else:
    mad_track = mad

mad_collider = mad
del(mad)

# Twiss machine to track
twiss_dfs, other_data = ost.twiss_and_check(mad_track, sequences_to_check,
        tol_beta=tol_beta, tol_sep=tol_sep,
        twiss_fname='twiss_track_intermediate',
        save_twiss_files=save_intermediate_twiss,
        check_betas_at_ips=check_betas_at_ips, check_separations_at_ips=False)


#####################
# Install bb lenses #
#####################

# Python approach
if enable_bb_python:
    if track_from_b4_mad_instance:
        bb_df_track = bb_dfs['b4']
        assert(sequence_to_track=='lhcb2')
    else:
        bb_df_track = bb_dfs['b1']
        assert(sequence_to_track=='lhcb1')

    pm.install_lenses_in_sequence(mad_track, bb_df_track, sequence_to_track)

    # Disable bb (to be activated later)
    mad_track.globals.on_bb_charge = 0
else:
    bb_df_track = None

# Legacy bb macros
if enable_bb_legacy:
    bbconfig = configuration['beambeam_config']
    assert(beam_to_configure == 1)
    assert(not(track_from_b4_mad_instance))
    assert(not(enable_bb_python))
    mad_track.globals['par_on_bb_switch'] = 1
    mad_track.set_variables_from_dict(
       params=configuration['pars_for_legacy_bb_macros'])
    mad_track.set_variables_from_dict(
            params={f'par_nho_ir{ir}': bbconfig['numberOfHOSlices']
            for ir in [1,2,5,8]})
    mad_track.input("call, file='modules/module_03_beambeam.madx';")


#########################
# Install crab cavities #
#########################

if enable_crabs:
    mad_track.input("call, file='optics_toolkit/enable_crabcavities.madx';")
    # They are left off, they will be swiched on at the end:
    mad_track.globals.on_crab1 = 0
    mad_track.globals.on_crab5 = 0


##############################################
# Save references for tuning and corrections #
##############################################

mad_track.input("call, file='modules/submodule_04_1b_save_references.madx';")


#####################
# Force on_disp = 0 #
#####################

mad_track.globals.on_disp = 0.
# will be restored later


#############
# Final use #
#############

mad_track.use(sequence_to_track)
# Disable use
mad_track._use = mad_track.use
mad_track.use = None


##############################
# Install and correct errors #
##############################

if enable_imperfections:
    mad_track.set_variables_from_dict(
            configuration['pars_for_imperfections'])
    mad_track.input("call, file='modules/module_04_errors.madx';")
else:
    # Synthesize knobs
    mad_track.input('call, file="modules/submodule_04a_s1_prepare_nom_twiss_table.madx";')
    if configuration['enable_knob_synthesis']:
        mad_track.input('exec, crossing_disable;')
        mad_track.input("call, file='modules/submodule_04e_s1_synthesize_knobs.madx';")
    mad_track.input('exec, crossing_restore;')


##################
# Machine tuning #
##################

# Enable bb for matchings
if match_q_dq_with_bb:
    mad_track.globals['on_bb_charge'] = 1
else:
    mad_track.globals['on_bb_charge'] = 0

# Switch on octupoles
brho = mad_track.globals.nrj*1e9/mad_track.globals.clight
i_oct = configuration['oct_current']
beam_str = {'lhcb1':'b1', 'lhcb2':'b2'}[sequence_to_track]
for ss in '12 23 34 45 56 67 78 81'.split():
   mad_track.input(f'kof.a{ss}{beam_str} = kmax_mo*({i_oct})/imax_mo/({brho});')
   mad_track.input(f'kod.a{ss}{beam_str} = kmax_mo*({i_oct})/imax_mo/({brho});')


# Correct linear coupling
qx_fractional, qx_integer = np.modf(configuration['qx0'])
qy_fractional, qy_integer = np.modf(configuration['qy0'])
coupl_corr_info = pm.coupling_correction(mad_track,
        n_iterations=configuration['N_iter_coupling'],
        qx_integer=qx_integer, qy_integer=qy_integer,
        qx_fractional=qx_fractional, qy_fractional=qy_fractional,
        tune_knob1_name=knob_names['qknob_1'][sequence_to_track],
        tune_knob2_name=knob_names['qknob_2'][sequence_to_track],
        cmr_knob_name=knob_names['cmrknob'][sequence_to_track],
        cmi_knob_name=knob_names['cmiknob'][sequence_to_track],
        sequence_name=sequence_to_track, skip_use=True)

# Add custom values to coupling knobs
mad_track.globals[knob_names['cmrknob'][sequence_to_track]] += configuration['delta_cmr']
mad_track.globals[knob_names['cmiknob'][sequence_to_track]] += configuration['delta_cmi']

# Check strength limits
if enable_imperfections:
    mad_track.input('call, file="errors/HL-LHC/corr_limit.madx";')

# Rematch the orbit at IPs
mad_track.input("call, file='tools/rematchCOIP.madx';")

# Rematch the CO in the arc for dispersion correction
if mad_track.globals.on_disp != 0:
    mad_track.input("call, file='tools/rematchCOarc.madx';")

# Match tunes and chromaticities
pm.match_tune_and_chromaticity(mad_track,
        q1=configuration['qx0'],
        q2=configuration['qy0'],
        dq1=configuration['chromaticity_x'],
        dq2=configuration['chromaticity_y'],
        tune_knob1_name=knob_names['qknob_1'][sequence_to_track],
        tune_knob2_name=knob_names['qknob_2'][sequence_to_track],
        chromaticity_knob1_name=knob_names['chromknob_1'][sequence_to_track],
        chromaticity_knob2_name=knob_names['chromknob_2'][sequence_to_track],
        sequence_name=sequence_to_track, skip_use=True)

# Check strength limits
if enable_imperfections:
    mad_track.input("call, file='errors/HL-LHC/corr_value_limit.madx';")

# Switch on bb lenses
mad_track.globals.on_bb_charge = 1.

# Switch on RF cavities
mad_track.globals['vrf400'] = configuration['vrf_total']
if sequence_to_track == 'lhcb1':
    mad_track.globals['lagrf400.b1'] = 0.5
elif sequence_to_track == 'lhcb2':
    mad_track.globals['lagrf400.b2'] = 0.

# Switch on crab cavities
if enable_crabs:
    mad_track.globals.on_crab1 = knob_settings['on_crab1']
    mad_track.globals.on_crab5 = knob_settings['on_crab5']


#####################
# Generate sixtrack #
#####################

if enable_bb_legacy:
    mad_track.input("call, file='modules/module_06_generate.madx'")
else:
    pm.generate_sixtrack_input(mad_track,
        seq_name=sequence_to_track,
        bb_df=bb_df_track,
        output_folder='./',
        reference_num_particles_sixtrack=(
            mad_track.sequence[sequence_to_track].beam.npart),
        reference_particle_charge_sixtrack=mad_track.sequence[sequence_to_track].beam.charge,
        emitnx_sixtrack_um=(
            mad_track.sequence[sequence_to_track].beam.exn),
        emitny_sixtrack_um=(
            mad_track.sequence[sequence_to_track].beam.eyn),
        sigz_sixtrack_m=(
            mad_track.sequence[sequence_to_track].beam.sigt),
        sige_sixtrack=(
            mad_track.sequence[sequence_to_track].beam.sige),
        ibeco_sixtrack=1,
        ibtyp_sixtrack=0,
        lhc_sixtrack=2,
        ibbc_sixtrack=0,
        radius_sixtrack_multip_conversion_mad=0.017,
        skip_mad_use=True)


#######################################
# Save optics and orbit at start ring #
#######################################

optics_and_co_at_start_ring_from_madx = pm.get_optics_and_orbit_at_start_ring(
        mad_track, sequence_to_track, skip_mad_use=True)
with open('./optics_orbit_at_start_ring_from_madx.json', 'w') as fid:
    json.dump(optics_and_co_at_start_ring_from_madx, fid, cls=pm.JEncoder)

########################
# Generate xtrack line #
########################
if enable_bb_legacy:
    print('xtrack line is not generated with bb legacy macros')
else:
    pm.generate_xsuite_line(mad_track, sequence_to_track, bb_df_track,
                    optics_and_co_at_start_ring_from_madx,
                    folder_name = './xsuite_lines',
                    skip_mad_use=True,
                    prepare_line_for_xtrack=True)

###################################
#         Save final twiss        #
###################################

mad_track.globals.on_bb_charge = 0
mad_track.twiss()
tdf = mad_track.get_twiss_df('twiss')
sdf = mad_track.get_summ_df('summ')
tdf.to_parquet('final_twiss_BBOFF.parquet')
sdf.to_parquet('final_summ_BBOFF.parquet')


mad_track.globals.on_bb_charge = 1
mad_track.twiss()
tdf = mad_track.get_twiss_df('twiss')
sdf = mad_track.get_summ_df('summ')
tdf.to_parquet('final_twiss_BBON.parquet')
sdf.to_parquet('final_summ_BBON.parquet')

#############################    
#  Save sequence and errors #
#############################
# N.B. this erases the errors in the mad_track instance
pm.save_mad_sequence_and_error(mad_track, sequence_to_track, filename='final')
