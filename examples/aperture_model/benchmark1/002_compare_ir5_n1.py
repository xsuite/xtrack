import matplotlib.pyplot as plt
import numpy as np
from pyoptics import BeamEnvelope

import xobjects as xo
import xtrack as xt
from xtrack.aperture import Aperture


context = xo.ContextCpu(omp_num_threads="auto")

lhc_with_metadata = xt.load("./lhc_aperture.json")
b1 = lhc_with_metadata["b1"]

aperture_model = Aperture.from_line_with_madx_metadata(
    b1,
    num_profile_points=100,
    include_offsets=True,
    context=context,
)

emittance_norm = 2.5e-6
apbbeat = 1.1
DParcx = 0.10
DParcy = 0.10
COmax = 0.002
dPmax = 0.0002
VMAXI = 30
SPECIF = 7

aperture_model.halo_params.update(
    {
        "emitx_norm": emittance_norm,
        "emity_norm": emittance_norm,
        "delta_rms": dPmax,
        "tol_co": COmax,
        "tol_disp": DParcx,  # MADX has different settings for x/y
        "tol_disp_ref_dx": 2.086,
        "tol_disp_ref_beta": 170.25,
        "tol_energy": 0.0,  # TO CHECK
        "tol_beta_beating": apbbeat,  # MADX has different settings for x/y
    }
)

ap = BeamEnvelope.from_apname("temp/ap_ir5b1.tfs")

s_positions = np.array(ap.ap.s, dtype=float)
n1_pyoptics = np.array(ap.ap.n1, dtype=float)

sigmas_rays, twiss, _, _ = aperture_model.get_aperture_sigmas_at_s(
    s_positions=s_positions,
    envelopes_num_points=36,
    method="rays",
)
sigmas_bisection, _, _, _ = aperture_model.get_aperture_sigmas_at_s(
    s_positions=s_positions,
    num_rays=360,
    method="bisection",
)
sigmas_exact, _, _, _ = aperture_model.get_aperture_sigmas_at_s(
    s_positions=s_positions,
    num_rays=360,
    method="exact",
)

n1_pyoptics_plot = np.where(n1_pyoptics > 9e5, np.inf, n1_pyoptics)
mask_rays = np.isfinite(n1_pyoptics_plot) & np.isfinite(sigmas_rays)
mask_bisection = np.isfinite(n1_pyoptics_plot) & np.isfinite(sigmas_bisection)
mask_exact = np.isfinite(n1_pyoptics_plot) & np.isfinite(sigmas_exact)

fig, (ax_top, ax_middle, ax_bottom) = plt.subplots(
    3,
    1,
    sharex=True,
    figsize=(10, 9),
    height_ratios=[3, 2, 1],
)

ax_top.plot(s_positions, n1_pyoptics_plot, label="n1 (MAD-X/pyoptics)", lw=1.5)
ax_top.plot(s_positions, sigmas_rays, label="n1 (Xtrack rays)", linestyle="--", lw=1.5)
ax_top.plot(s_positions, sigmas_bisection, label="n1 (Xtrack bisection)", linestyle=":", lw=1.5)
ax_top.plot(s_positions, sigmas_exact, label="n1 (Xtrack exact)", linestyle="-.", lw=1.5)
ax_top.set_ylabel(r"max beam size [$\sigma$]")
ax_top.set_title("IR5 B1 aperture comparison")
ax_top.legend()
ax_top.grid(True)

ax_middle.plot(s_positions, twiss.betx, label=r"$\beta_x$", lw=1.2)
ax_middle.plot(s_positions, twiss.bety, label=r"$\beta_y$", lw=1.2)
ax_middle.plot(s_positions, twiss.x * 1e3, label=r"$x_{co}$ [mm]", lw=1.2)
ax_middle.plot(s_positions, twiss.y * 1e3, label=r"$y_{co}$ [mm]", lw=1.2)
ax_middle.set_ylabel("Twiss / CO")
ax_middle.grid(True)
ax_middle.legend()

ax_bottom.plot(
    s_positions,
    sigmas_rays - n1_pyoptics_plot,
    lw=1.2,
    label=r"$\Delta n_1$ (rays - MAD-X)",
)
ax_bottom.plot(
    s_positions,
    sigmas_bisection - n1_pyoptics_plot,
    lw=1.2,
    label=r"$\Delta n_1$ (bisection - MAD-X)",
)
ax_bottom.plot(
    s_positions,
    sigmas_exact - n1_pyoptics_plot,
    lw=1.2,
    label=r"$\Delta n_1$ (exact - MAD-X)",
)
ax_bottom.axhline(0.0, color="k", lw=0.8, linestyle="--")
ax_bottom.set_xlabel("s [m]")
ax_bottom.set_ylabel(r"$\Delta n_1$")
ax_bottom.grid(True)
ax_bottom.legend()

plt.tight_layout()
plt.show()
