import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from lhcoptics import LHCOptics
from xdeps import Table
from xobjects import ContextCpu
from xtrack.aperture import Aperture


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
        line.metadata["aperture_offsets"][f"ip{ipn}"] = tfs._data.copy()

opt = LHCOptics.from_xsuite(lhc)
mad = opt.make_madx_model()

context = ContextCpu(omp_num_threads="auto")
b1 = lhc.b1
apx = Aperture.from_line_with_madx_metadata(
    b1,
    num_profile_points=100,
    include_offsets=True,
    context=context,
)

target_name = "mbrb.5l4.b1"
resolution = 0.1

xs_name, = b1.get_table().rows[f"{target_name}.*"].name

sigmas, twiss, cross_sections, max_envelope = apx.get_aperture_sigmas_at_element(
    element_name=xs_name,
    resolution=resolution,
    method="bisection",
)

cross_sections_ref, _ = apx.cross_sections_at_element(
    element_name=xs_name,
    resolution=resolution,
)

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
