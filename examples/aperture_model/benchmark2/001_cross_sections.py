import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from lhcoptics import LHCOptics
from xdeps import Table
from xobjects import ContextCpu


base = "./acc-models-lhc/"
lhc = xt.load(f"{base}/xsuite/lhc_aperture.json")
lhc.vars.load(f"{base}/strengths/cycle_round_v0/opt_6000.madx")
lhc.set_particle_ref(p0c=450e9)

lhc.b1.metadata["aperture_offsets"] = {}
lhc.b2.metadata["aperture_offsets"] = {}
for ipn in range(1, 9):
    for beam in "14":
        tfs = Table.from_tfs(f"./temp/offset.ip{ipn}.b{beam}.tfs")
        line = lhc.b1 if beam == "1" else lhc.b2
        offset_data = tfs._data.copy()
        offset_data['reversed'] = beam == '4'
        line.metadata["aperture_offsets"][f"ip{ipn}"] = offset_data

opt = LHCOptics.from_xsuite(lhc)
mad = opt.make_madx_model()

context = ContextCpu(omp_num_threads="auto")
b1 = lhc.b1
apx = b1.aperture.from_madx_metadata(
    num_profile_points=100,
    include_offsets=True,
    context=context,
)

target_name = "mbrb.5l4.b1"
resolution = 0.1

xs_name, = b1.get_table().rows[f"{target_name}.*"].name

n1_table, twiss = apx.get_aperture_sigmas_at_element(
    element_name=xs_name,
    resolution=resolution,
    method="bisection",
    output_cross_sections=True,
    output_max_envelopes=True,
)
sigmas = n1_table.n1
cross_sections = n1_table.cross_section
max_envelope = n1_table.envelope

cross_sections_table = apx.get_cross_sections_at_element(
    element_name=xs_name,
    resolution=resolution,
)
cross_sections_ref = cross_sections_table.cross_section

ap_centre = (np.min(cross_sections_ref, axis=1) + np.max(cross_sections_ref, axis=1)) / 2

plt.figure()
for pt, ct in zip(cross_sections_ref, ap_centre):
    plt.plot(
        pt[:, 0] - ct[0],
        pt[:, 1] - ct[1],
        c="gray",
        linestyle="--",
        label="aperture" if np.all(ct == ap_centre[0]) else "",
    )

for pt, ct in zip(max_envelope, ap_centre):
    plt.plot(
        pt[:, 0] - ct[0],
        pt[:, 1] - ct[1],
        c="cyan",
        linestyle=":",
        label="beam envelope" if np.all(ct == ap_centre[0]) else "",
    )

plt.gca().set_aspect("equal")
plt.xlabel("x [m] relative to aperture centre")
plt.ylabel("y [m] relative to aperture centre")
plt.suptitle(xs_name)
plt.title(
    f"bisection n1 min = {np.min(sigmas):.5f}, "
    f"s = [{twiss.s[0]:.3f}, {twiss.s[-1]:.3f}] m"
)
plt.legend()
plt.show()

print(f"Target element: {xs_name}")
print(f"s range: {twiss.s[0]:.6f} .. {twiss.s[-1]:.6f} m")
print(f"min sigma: {np.min(sigmas):.8f}")
print(f"max sigma: {np.max(sigmas):.8f}")
