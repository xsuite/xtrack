import matplotlib.pyplot as plt
import numpy as np

import xobjects as xo
import xtrack as xt

from xtrack.aperture import Aperture

context = xo.ContextCpu(omp_num_threads='auto')

lhc_with_metadata = xt.load('./benchmark1/lhc_aperture.json')
b1 = lhc_with_metadata['b1']
lhc_length = b1.get_length()

aperture_model = Aperture.from_line_with_madx_metadata(b1, num_profile_points=100, context=context)

end_s = lhc_length / 8
s_positions = np.linspace(0, end_s, int(end_s))

# Calculate n1 with the ``rays`` method
sig_rays, tw_rays, aper_rays, _ = aperture_model.get_aperture_sigmas_at_s(
    s_positions=np.linspace(0, lhc_length, int(lhc_length)),
    method='rays',
)

# Calculate extents
envel, tw_envel = aperture_model.get_envelope_at_s(
    s_positions=np.linspace(0, lhc_length, int(lhc_length)),
    sigmas=5,
    envelopes_num_points=12,
    include_aper_tols=False,
)

cs_s = np.linspace(0, lhc_length, int(lhc_length))
cs_table = aperture_model.cross_sections_at_s(s_positions=cs_s)
cs = cs_table.cross_section

# Calculate n1's with the ``bisection`` method
# sig_bisect, tw_bisect, aper_bisect, max_envelope = aperture_model.get_aperture_sigmas_at_s(
#     s_positions=np.linspace(0, lhc_length, int(lhc_length)),
#     method='bisection',
# )

# PLOT envelope sigmas
# plt.plot(tw_rays.s, sig_rays, label=r'min envelope [$\sigma$] (rays)')

# plt.plot(tw_bisect.s, sig_bisect, label=r'max envelope [$\sigma$] (bisection)', linestyle='--')

# plt.plot(tw_rays.s, np.max(aper_rays[:, :, 0], axis=1), label=r'+ horizontal extent [mm]')
# plt.plot(tw_rays.s, np.min(aper_rays[:, :, 0], axis=1), label=r'- horizontal extent [mm]')
# plt.plot(tw_rays.s, np.max(aper_rays[:, :, 1], axis=1), label=r'+ vertical extent [mm]')
# plt.plot(tw_rays.s, np.min(aper_rays[:, :, 1], axis=1), label=r'- vertical extent [mm]')

# plt.legend()
# plt.show()
#
# plt.scatter(tw_rays.x, tw_rays.y)


fig, (ax, ay) = plt.subplots(2, sharex=True)
fig.suptitle(r'Interpolated apertures and beam at 3$\sigma$')

min_envel_x = np.min(envel[:, :, 0], axis=1) * 1000
max_envel_x = np.max(envel[:, :, 0], axis=1) * 1000
min_aper_x = np.min(cs[:, :, 0], axis=1) * 1000
max_aper_x = np.max(cs[:, :, 0], axis=1) * 1000

ax.fill_between(tw_envel.s, min_envel_x, max_envel_x, color='b', alpha=0.3)
ax.plot(cs_s, min_aper_x, color='k', marker='.')
ax.plot(cs_s, max_aper_x, color='k', marker='.')
ax.set_ylabel(r'horizontal aperture [mm]')
ax.set_ylim([-100, 100])

# ax_sig = ax.twinx()
# ax_sig.plot(tw_rays.s, sig_rays[:, 0], label=r'horizonal envelope [$\sigma$] (rays)', color='pink')
# ax_sig.set_ylabel(r'horizontal n1 [$sigma$]')

min_envel_y = np.min(envel[:, :, 1], axis=1) * 1000
max_envel_y = np.max(envel[:, :, 1], axis=1) * 1000
min_aper_y = np.min(cs[:, :, 1], axis=1) * 1000
max_aper_y = np.max(cs[:, :, 1], axis=1) * 1000
ay.set_ylabel(r'vertical aperture [mm]')
ay.set_ylim([-100, 100])

ay.fill_between(tw_envel.s, min_envel_y, max_envel_y, color='r', alpha=0.3)
ay.plot(cs_s, min_aper_y, color='k', marker='.')
ay.plot(cs_s, max_aper_y, color='k', marker='.')

# ay_sig = ay.twinx()
# ay_sig.plot(tw_rays.s, sig_rays[:, 1], label=r'vertical envelope [$\sigma$] (rays)', color='violet')
# ay_sig.set_ylabel(r'horizontal n1 [$sigma$]')

ay.set_xlabel(r's [m]')
plt.show()
