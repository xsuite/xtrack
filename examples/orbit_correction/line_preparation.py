from scipy.constants import c as clight

def rename_coupling_knobs_and_coefficients(line, beamn):

    line.vars[f'c_minus_re_b{beamn}'] = 0
    line.vars[f'c_minus_im_b{beamn}'] = 0
    for ii in [1, 2, 3, 4, 5, 6, 7, 8]:
        for jj, nn in zip([1, 2], ['re', 'im']):
            old_name = f'b{ii}{jj}'
            new_name = f'coeff_skew_{ii}{jj}_b{beamn}'

            # Copy value in new variable
            line.vars[new_name] = line.vars[old_name]._value

            # Zero old variable
            line.vars[old_name] = 0

            # Identify controlled circuit
            targets = line.vars[old_name]._find_dependant_targets()
            if len(targets) > 1: # Controls something
                ttt = [t for t in targets if repr(t).startswith('vars[') and
                    repr(t) != f"vars['{old_name}']"]
                assert len(ttt) > 0
                assert len(ttt) < 3

                for kqs_knob in ttt:
                    kqs_knob_str = repr(kqs_knob)
                    assert "'" in kqs_knob_str
                    assert '"' not in kqs_knob_str
                    var_name = kqs_knob_str.split("'")[1]
                    assert var_name.startswith('kqs')
                    line.vars[var_name] += (line.vars[new_name]
                                * line.vars[f'c_minus_{nn}_b{beamn}'])

def define_octupole_current_knobs(line, beamn):
    line.vars[f'p0c_b{beamn}'] = line.particle_ref.p0c[0]
    line.vars[f'q0_b{beamn}'] = line.particle_ref.q0
    line.vars[f'brho0_b{beamn}'] = (line.vars[f'p0c_b{beamn}']
                                / line.vars[f'q0_b{beamn}'] / clight)

    line.vars[f'i_oct_b{beamn}'] = 0
    for ss in '12 23 34 45 56 67 78 81'.split():
        line.vars[f'kof.a{ss}b{beamn}'] = (
            line.vars['kmax_mo']
            * line.vars[f'i_oct_b{beamn}'] / line.vars['imax_mo']
            / line.vars[f'brho0_b{beamn}'])
        line.vars[f'kod.a{ss}b{beamn}'] = (
            line.vars['kmax_mo']
            * line.vars[f'i_oct_b{beamn}'] / line.vars['imax_mo']
            / line.vars[f'brho0_b{beamn}'])

def add_correction_term_to_dipole_correctors(line):
    # Add correction term to all dipole correctors
    line.vars['on_corr_co'] = 1
    for kk in list(line.vars._owner.keys()):
        if kk.startswith('acb'):
            line.vars['corr_co_'+kk] = 0
            line.vars[kk] += (line.vars['corr_co_'+kk]
                                * line.vars['on_corr_co'])