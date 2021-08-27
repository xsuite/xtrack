import numpy as np

import sixtracktools
import xline

six = sixtracktools.SixInput(".")
line = xline.Line.from_sixinput(six)
iconv = line.other_info["iconv"]

sixdump = sixtracktools.SixDump101("res/dump3.dat")[1::2]


def compare(prun, pbench):
    out = []
    for att in "x px y py zeta delta".split():
        vrun = getattr(prun, att)
        vbench = getattr(pbench, att)
        diff = vrun - vbench
        out.append(abs(diff))
        print(f"{att:<5} {vrun:22.13e} {vbench:22.13e} {diff:22.13g}")
    print(f"max {max(out):21.12e}")
    return max(out), out


print("")
diffs = []
s_coord = []
for ii in range(1, len(iconv)):
    jja = iconv[ii - 1]
    jjb = iconv[ii]
    prun = xline.Particles(**sixdump[ii - 1].get_minimal_beam())
    print(f"\n-----sixtrack={ii} xline={jja} --------------")
    # print(f"pysixtr {jja}, x={prun.x}, px={prun.px}")
    for jj in range(jja + 1, jjb + 1):
        label, elem = line.element_names[jj], line.elements[jj]
        elem.track(prun)
        print(f"{jj} {label},{str(elem)[:50]}")
    pbench = xline.Particles(**sixdump[ii].get_minimal_beam())
    s_coord.append(pbench.s)
    # print(f"sixdump {ii}, x={pbench.x}, px={pbench.px}")
    print("-----------------------")
    out, out_all = compare(prun, pbench)
    print("-----------------------\n\n")
    diffs.append(out_all)
    if out > 1e-13:
        print("Too large discrepancy")
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
