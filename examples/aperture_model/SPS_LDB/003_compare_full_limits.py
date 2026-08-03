import matplotlib.pyplot as plt
import xtrack as xt

S_TOL = 3.1e-2

sps = xt.load('sps.json')
env = xt.load('sps_env.json')
sps_with_limits = env.lines['sps_with_limits']

aperture = xt.Aperture.from_json(
    'sps_aperture.json',
    line=sps,
    s_tol=S_TOL,
    _skip_validity_check=True,
)

aperture_from_limits = xt.Aperture.from_line_with_limits(sps_with_limits, s_tol=S_TOL)

s_positions = aperture.s_around_transitions(resolution=0.1)
sections_original = aperture.cross_sections_at_s(s_positions, extents=True, polygons=False)
sections_recovered = aperture_from_limits.cross_sections_at_s(s_positions, extents=True, polygons=False)

fig, (ax_x, ax_y) = plt.subplots(2, 1, sharex=True)

ax_x.plot(s_positions, sections_original.min_x, color='C0', linestyle='-', label='original min')
ax_x.plot(s_positions, sections_original.max_x, color='C0', linestyle='-', label='original max')
ax_x.plot(s_positions, sections_recovered.min_x, color='C1', linestyle='--', label='from limits min')
ax_x.plot(s_positions, sections_recovered.max_x, color='C1', linestyle='--', label='from limits max')
ax_x.set_ylabel('x [m]')
ax_x.legend()

ax_y.plot(s_positions, sections_original.min_y, color='C0', linestyle='-', label='original min')
ax_y.plot(s_positions, sections_original.max_y, color='C0', linestyle='-', label='original max')
ax_y.plot(s_positions, sections_recovered.min_y, color='C1', linestyle='--', label='from limits min')
ax_y.plot(s_positions, sections_recovered.max_y, color='C1', linestyle='--', label='from limits max')
ax_y.set_ylabel('y [m]')
ax_y.set_xlabel('s [m]')
ax_y.legend()

fig.suptitle('SPS aperture extents: original model vs recovered from limits')
plt.show()
