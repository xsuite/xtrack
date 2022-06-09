# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import pymask as pm

# The parts marked by (*) in following need to be
# adapted according to knob definitions

def build_sequence(mad, beam):

    # Select beam
    mad.input(f'mylhcbeam = {beam}')

    # Make link to optics toolkit
    pm.make_links({'optics_toolkit': 'optics_repository/HLLHCV1.4/toolkit'},
        force=True)

    mad.input('''

        ! Specify machine version
        ver_lhc_run = 0;
        ver_hllhc_optics = 1.4;

        ! Get the toolkit
        call, file="optics_repository/HLLHCV1.4/toolkit/macro.madx";


        ! Build sequence
        option, -echo,-warn,-info;
        if (mylhcbeam==4){
          call,file="optics_repository/runIII/lhcb4.seq";
        } else {
          call,file="optics_repository/runIII/lhc.seq";
        };
        option, -echo, warn,-info;

        !Install HL-LHC
        call, file="optics_repository/HLLHCV1.4/hllhc_sequence.madx";


        ! Slice nominal sequence
        exec, myslice;

        ! Install placeholder elements for errors (set to zero)
        call, file="errors/HL-LHC/install_MQXF_fringenl.madx";    ! adding fringe place holder
        call, file="errors/HL-LHC/install_MCBXFAB_errors.madx";   ! adding D1 corrector placeholders in IR1/5 (for errors)
        call, file="errors/HL-LHC/install_MCBRD_errors.madx";     ! adding D2 corrector placeholders in IR1/5 (for errors)
        call, file="errors/HL-LHC/install_NLC_errors.madx";       ! adding non-linear corrector placeholders in IR1/5 (for errors)

        !Cycling w.r.t. to IP3 (mandatory to find closed orbit in collision in the presence of errors)
        if (mylhcbeam<3){
          seqedit, sequence=lhcb1; flatten; cycle, start=IP3; flatten; endedit;
        };
        seqedit, sequence=lhcb2; flatten; cycle, start=IP3; flatten; endedit;
        ''')

def apply_optics(mad, optics_file):
    mad.call(optics_file)


def set_optics_specific_knobs(mad, knob_settings, mode=None):

    # Copy knob settings to mad variable space
    mad.set_variables_from_dict(params=knob_settings)

    # A check
    if mad.globals.nrj < 500:
        assert knob_settings['on_disp'] == 0

    # A knob redefinition
    mad.input('on_alice := on_alice_normalized * 7000./nrj;')
    mad.input('on_lhcb := on_lhcb_normalized * 7000./nrj;')


def twiss_and_check(mad, sequences_to_check, twiss_fname,
        tol_beta=1e-3, tol_sep=1e-6, save_twiss_files=True,
        check_betas_at_ips=True, check_separations_at_ips=True):

    var_dict = mad.get_variables_dicts()
    twiss_dfs = {}
    summ_dfs = {}
    for ss in sequences_to_check:
        mad.use(ss)
        mad.twiss()
        tdf = mad.get_twiss_df('twiss')
        twiss_dfs[ss] = tdf
        sdf = mad.get_summ_df('summ')
        summ_dfs[ss] = sdf

    if save_twiss_files:
        for ss in sequences_to_check:
            tt = twiss_dfs[ss]
            if twiss_fname is not None:
                tt.to_parquet(twiss_fname + f'_seq_{ss}.parquet')

    if check_betas_at_ips:
        for ss in sequences_to_check:
            tt = twiss_dfs[ss]
            _check_beta_at_ips_against_madvars(beam=ss[-1],
                    twiss_df=tt,
                    variable_dicts=var_dict,
                    tol=tol_beta)
        print('IP beta test against knobs passed!')

    if check_separations_at_ips:
        twiss_df_b1 = twiss_dfs['lhcb1']
        twiss_df_b2 = twiss_dfs['lhcb2']
        _check_separations_at_ips_against_madvars(twiss_df_b1, twiss_df_b2,
                var_dict, tol=tol_sep)
        print('IP separation test against knobs passed!')

    other_data = {}
    other_data.update(var_dict)
    other_data['summ_dfs'] = summ_dfs

    return twiss_dfs, other_data

def lumi_control(mad, twiss_dfs, configuration, knob_names):
    from scipy.optimize import least_squares

    # Leveling in IP8
    sep_plane_ip8 = configuration['sep_plane_ip8']
    sep_knobname_ip8 = knob_names['sepknob_ip8_mm']

    L_target_ip8 = configuration['lumi_ip8']
    def function_to_minimize_ip8(sep8_m):
        my_dict_IP8=pm.get_luminosity_dict(
            mad, twiss_dfs, 'ip8', configuration['nco_IP8'])
        my_dict_IP8[sep_plane_ip8 + '_1']=np.abs(sep8_m)
        my_dict_IP8[sep_plane_ip8 + '_2']=-np.abs(sep8_m)
        return np.abs(pm.luminosity(**my_dict_IP8) - L_target_ip8)
    sigma_sep_b1_ip8=np.sqrt(twiss_dfs['lhcb1'].loc['ip8:1']['bet'+sep_plane_ip8]
               * mad.sequence.lhcb1.beam['e'+sep_plane_ip8])
    optres_ip8=least_squares(function_to_minimize_ip8, sigma_sep_b1_ip8)
    mad.globals[sep_knobname_ip8] = (np.sign(mad.globals[sep_knobname_ip8])
                                * np.abs(optres_ip8['x'][0])*1e3)

    # Halo collision in IP2
    sep_plane_ip2 = configuration['sep_plane_ip2']
    sep_knobname_ip2 = knob_names['sepknob_ip2_mm']
    sigma_sep_b1_ip2=np.sqrt(
            twiss_dfs['lhcb1'].loc['ip2:1']['bet'+sep_plane_ip2]
            * mad.sequence.lhcb1.beam['e'+sep_plane_ip2])
    mad.globals[sep_knobname_ip2] = (np.sign(mad.globals[sep_knobname_ip2])
            * configuration['fullsep_in_sigmas_ip2']*sigma_sep_b1_ip2/2*1e3)


def _check_beta_at_ips_against_madvars(beam, twiss_df, variable_dicts, tol):
    twiss_value_checks=[]
    for iip, ip in enumerate([1,2,5,8]):
        for plane in ['x', 'y']:
            # (*) Adapet based on knob definitions
            twiss_value_checks.append({
                    'element_name': f'ip{ip}:1',
                    'keyword': f'bet{plane}',
                    'varname': f'bet{plane}ip{ip}b{beam}',
                    'tol': tol[iip]})

    pm.check_twiss_against_madvars(twiss_value_checks, twiss_df, variable_dicts)

def _check_separations_at_ips_against_madvars(twiss_df_b1, twiss_df_b2,
        variables_dict, tol):

    separations_to_check = []
    for iip, ip in enumerate([1,2,5,8]):
        for plane in ['x', 'y']:
            # (*) Adapet based on knob definitions
            separations_to_check.append({
                    'element_name': f'ip{ip}:1',
                    'scale_factor': -2*1e-3,
                    'plane': plane,
                    # knobs like on_sep1h, onsep8v etc
                    'varname': f'on_sep{ip}'+{'x':'h', 'y':'v'}[plane],
                    'tol': tol[iip]})
    pm.check_separations_against_madvars(separations_to_check,
            twiss_df_b1, twiss_df_b2, variables_dict)
