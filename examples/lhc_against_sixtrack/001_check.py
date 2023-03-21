# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import sixtracktools
import xtrack as xt
import xobjects as xo
import xpart as xp
import xfields as xf

# sixtrack_folder = './sixtrack_lhc_no_bb/res_onmom';
# atol = {
#     'x':1e-16, 'px':1e-16, 'y':1e-16, 'py':1e-16, 'zeta':6e-14, 'delta':1e-15}

#sixtrack_folder = './sixtrack_lhc_no_bb/res_offmom';
#atol = {
#    'x':1e-16, 'px':1e-16, 'y':1e-16, 'py':1e-16, 'zeta':5e-14, 'delta':1e-15}

sixtrack_folder = '../../test_data/hllhc_14/res'; atol = 5e-12
atol = {
    'x':1e-15, 'px':2e-12, 'y':1e-15, 'py':2e-12, 'zeta':2e-14, 'delta':1e-15}

context = xo.ContextCpu()

sixinput = sixtracktools.SixInput(sixtrack_folder)
line = sixinput.generate_xtrack_line()
line.build_tracker(_context=context)
iconv = line.other_info["iconv"]

sixdump_all = sixtracktools.SixDump101(sixtrack_folder + "/dump3.dat")

if any(ee.__class__.__name__.startswith('BeamBeam') for ee in line.elements):
    Nele_st = len(iconv)
    sixdump_CO = sixdump_all[::2][:Nele_st]
    # Get closed-orbit from sixtrack 
    p0c_eV = sixinput.initialconditions[-3] * 1e6
    part_on_CO = xp.Particles(
            p0c=p0c_eV,
            x=sixdump_CO.x[0],
            px=sixdump_CO.px[0],
            y=sixdump_CO.y[0],
            py=sixdump_CO.py[0],
            zeta=sixdump_CO.tau[0]*sixdump_CO.beta0[0],
            delta=sixdump_CO.delta[0])
    xf.configure_orbit_dependent_parameters_for_bb(line,
                           particle_on_co=part_on_CO)

def compare(prun, pbench):
    out = []
    for att in "x px y py zeta delta".split():
        vrun = getattr(prun, att)[0]
        vbench = getattr(pbench, att)[0]
        diff = vrun - vbench
        out.append(abs(diff))
        print(f"{att:<5} {vrun:22.13e} {vbench:22.13e} {diff:22.13g}")
    print(f"max {max(out):21.12e}")
    return max(out), out

sixdump = sixdump_all[1::2]
print("")
diffs = []
s_coord = []
x_run = []
x_bench = []
s_run = []
s_bench = []
for ii in range(1, len(iconv)):
    jja = iconv[ii - 1]
    jjb = iconv[ii]
    prun = xp.Particles.from_dict(_context=context,
            dct=sixdump[ii - 1].get_minimal_beam())
    prun.state[0]=1
    prun.reorganize()
    print(f"\n-----sixtrack={ii} xtrack={jja} --------------")
    # print(f"pysixtr {jja}, x={prun.x}, px={prun.px}")
    for jj in range(jja + 1, jjb + 1):
        label, elem = line.element_names[jj], line.elements[jj]
        #elem.track(prun)
        line.track(particles=prun, ele_start=jj, num_elements=1)
        print(f"{jj} {label},{str(elem)[:50]}")
    pbench = xp.Particles.from_dict(sixdump[ii].get_minimal_beam())
    s_coord.append(pbench.s)
    x_run.append(prun.x)
    x_bench.append(pbench.x)
    s_run.append(prun.s)
    s_bench.append(pbench.s)
    # print(f"sixdump {ii}, x={pbench.x}, px={pbench.px}")
    print("-----------------------")
    out, out_all = compare(prun, pbench)
    print("-----------------------\n\n")
    diffs.append(out_all)
    issue_found = False
    for ii, nn in enumerate('x px y py zeta delta'.split()):
        if out_all[ii] > atol[nn]:
            print("Too large discrepancy")
            issue_found = True

    if issue_found:
        break

diffs = np.array(diffs)

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1, figsize=(6.4*1.5, 4.8*1.3))
for ii, (vv, uu) in enumerate(
        zip(['x', 'px', 'y', 'py', r'$\zeta$', r'$\delta$'],
            ['[m]', '[-]', '[m]', '[-]', '[m]', '[-]'])):
    ax = fig.add_subplot(3, 2, ii+1)
    ax.plot(s_coord, diffs[:, ii])
    ax.set_ylabel('Difference on '+ vv + ' ' + uu)
    ax.set_xlabel('s [m]')
fig.subplots_adjust(hspace=.48)


plt.show()
