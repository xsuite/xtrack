from pathlib import Path

import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt

context = xo.ContextCpu(omp_num_threads='auto')

lhc_with_metadata = xt.load(Path(__file__).with_name('lhc_aperture.json'))
b1 = lhc_with_metadata['b1']
lhc_length = b1.get_length()

# This older data file predates the explicit offset-frame direction flag.
for section in b1.metadata['aperture_offsets'].values():
    section['reversed'] = False

aperture_model = b1.aperture.from_madx_metadata(num_profile_points=100, context=context)

mqxfa_name = 'mqy.4r1.b1'

# Calculate n1 with the ``rays`` method
n1_rays, tw_rays = aperture_model.get_aperture_sigmas_at_element(
    element_name=mqxfa_name,
    resolution=0.1,
    method='rays',
    output_cross_sections=True,
)
sig_rays = n1_rays.n1
aper_rays = n1_rays.cross_section

hvd_rays, _ = aperture_model.get_hvd_aperture_sigmas_at_element(
    element_name=mqxfa_name,
    resolution=0.1,
)

# Calculate n1's with the ``bisection`` method
n1_bisect, tw_bisect = aperture_model.get_aperture_sigmas_at_element(
    element_name=mqxfa_name,
    resolution=0.1,
    method='bisection',
    output_cross_sections=True,
    output_max_envelopes=True,
)
sig_bisect = n1_bisect.n1
aper_bisect = n1_bisect.cross_section
max_envelope = n1_bisect.envelope

# Get envelope at arbitrary sigma
envelopes_table, tw_envel = aperture_model.get_envelope_at_element(
    element_name=mqxfa_name,
    resolution=0.1,
    sigmas=1,
)
envelopes = envelopes_table.cross_section

aper_table = aperture_model.get_cross_sections_at_element(
    element_name=mqxfa_name,
    resolution=0.1,
)
aper_envel = aper_table.cross_section

# PLOT envelope sigmas
plt.plot(tw_rays.s, hvd_rays.n1_horizontal, label=r'horizonal envelope [$\sigma$] (rays)')
plt.plot(tw_rays.s, hvd_rays.n1_vertical, label=r'vertical envelope [$\sigma$] (rays)')
plt.plot(tw_rays.s, hvd_rays.n1_diagonal, label=r'diagonal envelope [$\sigma$] (rays)')
plt.plot(tw_rays.s, sig_rays, label=r'min envelope [$\sigma$] (rays)', linestyle=':')

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
