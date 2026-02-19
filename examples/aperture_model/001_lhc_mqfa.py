import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt

from xtrack.aperture import Aperture

context = xo.ContextCpu(omp_num_threads='auto')

lhc_with_metadata = xt.load('./lhc_aperture.json')
b1 = lhc_with_metadata['b1']
lhc_length = b1.get_length()

aperture_model = Aperture.from_line_with_madx_metadata(b1, line_name='b1', context=context)

mqxfa_name = 'mqy.4r1.b1'

# Calculate n1 with the ``rays`` method
sig_rays, tw_rays, aper_rays, _ = aperture_model.get_aperture_sigmas_at_element(
    line_name="b1",
    element_name=mqxfa_name,
    resolution=0.1,
    cross_sections_num_points=100,
    method='rays',
)

# Calculate n1's with the ``bisection`` method
sig_bisect, tw_bisect, aper_bisect, max_envelope = aperture_model.get_aperture_sigmas_at_element(
    line_name="b1",
    element_name=mqxfa_name,
    resolution=0.1,
    cross_sections_num_points=100,
    method='bisection',
)

# Get envelope at arbitrary sigma
envelopes, aper_envel, tw_envel = aperture_model.get_apertures_and_envelope_at_element(
    line_name='b1',
    element_name=mqxfa_name,
    resolution=0.1,
    sigmas=1,
)

# PLOT envelope sigmas
plt.plot(tw_rays.s, sig_rays[:, 0], label=r'horizonal envelope [$\sigma$] (rays)')
plt.plot(tw_rays.s, sig_rays[:, 1], label=r'vertical envelope [$\sigma$] (rays)')
plt.plot(tw_rays.s, sig_rays[:, 2], label=r'diagonal envelope [$\sigma$] (rays)')

plt.plot(tw_bisect.s, sig_bisect, label=r'max envelope [$\sigma$] (bisection)', linestyle='--')

plt.legend()
plt.show()

plt.scatter(tw_rays.x, tw_rays.y)

# PLOT max sigma beam in aperture
for pt in aper_rays:
    plt.plot(pt[:, 0], pt[:, 1], c='k')

for pt in aper_bisect:
    plt.plot(pt[:, 0], pt[:, 1], c='gray', linestyle='--')

for pt in max_envelope:
    plt.plot(pt[:, 0], pt[:, 1])

plt.gca().set_aspect('equal')
plt.legend()
plt.show()

# PLOT arbitrary sigma beam in aperture
for pt in aper_envel:
    plt.plot(pt[:, 0], pt[:, 1], c='k')

for pt in envelopes:
    plt.plot(pt[:, 0], pt[:, 1])

plt.gca().set_aspect('equal')
plt.legend()
plt.show()
