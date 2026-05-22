import xtrack as xt

env = xt.load('fccee_z_lcc_solenoid.json')
line = env.fccee_p_ring

# Turn off exprimental solenoids
line['on_sol_ipa'] = 0
line['on_sol_ipd'] = 0
line['on_sol_ipg'] = 0
line['on_sol_ipj'] = 0

# Turn off corrections
# (compens. solenoids, doublet tilts, orbit correction, optics correction)
line['on_sol_corr_ipa'] = 0
line['on_sol_corr_ipd'] = 0
line['on_sol_corr_ipg'] = 0
line['on_sol_corr_ipj'] = 0

tw_off = line.twiss6d()

# Turn on exprimental solenoids
line['on_sol_ipa'] = 1
line['on_sol_ipd'] = 1
line['on_sol_ipg'] = 1
line['on_sol_ipj'] = 1

# Turn on corrections
# (compens. solenoids, doublet tilts, orbit correction, optics correction)
line['on_sol_corr_ipa'] = 1
line['on_sol_corr_ipd'] = 1
line['on_sol_corr_ipg'] = 1
line['on_sol_corr_ipj'] = 1

tw = line.twiss6d()