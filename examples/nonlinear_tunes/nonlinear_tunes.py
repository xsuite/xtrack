import xtrack as xt

import detuning

nemitt_x = 1e-6
nemitt_y = 1e-6

line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7e12)
line.build_tracker()

for sec in [12, 23, 34, 45, 56, 67, 78, 81]:
    for tt in ['f', 'd']:
        line.vars[f"ko{tt}.a{sec}b1"] = 18

axx, axy, ayx, ayy = detuning.get_amplitude_detuning(line)

print(f"axx = {1.e-6*axx:.3f} um^-1")
print(f"axy = {1.e-6*axy:.3f} um^-1")
print(f"ayx = {1.e-6*ayx:.3f} um^-1")
print(f"ayy = {1.e-6*ayy:.3f} um^-1")
# expecting
# axx = 1.070 um^-1
# axy = -0.573 um^-1
# ayx = -0.514 um^-1
# ayy = 1.073 um^-1

order=3
chromaticity = detuning.get_nonlinear_chromaticity(line, order=order)

print()
for ii in range(2, order+1):
    print("Q" + ii *"'" + (order - ii)*" " + f"x = {chromaticity.qx_derivatives[ii]:.3e}")
for ii in range(2, order+1):
    print("Q" + ii *"'" + (order - ii)*" " + f"y = {chromaticity.qx_derivatives[ii]:.3e}")
