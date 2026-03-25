import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from lhcoptics import LHCOptics
from xdeps import Table
from xobjects import ContextCpu
from xtrack.aperture import Aperture

which_ir = '2'

base = "./acc-models-lhc/"

lhc = xt.load(f"{base}/xsuite/lhc_aperture.json")
lhc.vars.load(f"{base}/strengths/cycle_round_v0/opt_6000.madx")
lhc.set_particle_ref(p0c=450e9)

opt = LHCOptics.from_xsuite(lhc)
mad = opt.make_madx_model()
apm = mad.get_ap_irs()

lhc.b1.metadata["aperture_offsets"] = {}
lhc.b2.metadata["aperture_offsets"] = {}
for ipn in range(1, 9):
    for beam in "14":
        tfs = Table.from_tfs(f"./temp/offset.ip{ipn}.b{beam}.tfs")
        line = lhc.b1 if beam == "1" else lhc.b2
        line.metadata["aperture_offsets"][f"ip{ipn}"] = tfs._data.copy()

ir = apm[f"ir{which_ir}b1"]

context = ContextCpu(omp_num_threads="auto")
b1 = lhc.b1
aperture_model = Aperture.from_line_with_madx_metadata(
    b1,
    num_profile_points=100,
    include_offsets=True,
    context=context,
)

aperture_model.halo_params.update(
    {
        "emitx_norm": ir.exn,
        "emity_norm": ir.eyn,
        "delta_rms": ir.dp_bucket_size,
        "tol_co": ir.co_radius,
        "tol_disp": ir.paras_dx,
        "tol_disp_ref_dx": ir.dqf,
        "tol_disp_ref_beta": ir.betaqfx,
        "tol_energy": 0.0,
        "tol_beta_beating": ir.beta_beating,
    }
)

s_ip_m, = ir.rows[f"ip{which_ir}:1"].s
s_ip_x, = b1.get_table().rows[f"ip{which_ir}"].s
s_positions = np.array(ir.s - s_ip_m + s_ip_x, dtype=float)
n1_madx = np.array(ir.n1, dtype=float)

n1_rays, twiss = aperture_model.get_aperture_sigmas_at_s(
    s_positions=s_positions,
    method="rays",
)
sigmas_rays = n1_rays.n1
# n1_bisection, _ = aperture_model.get_aperture_sigmas_at_s(
#     s_positions=s_positions,
#     method="bisection",
#     envelopes_num_points=36,
# )
# sigmas_bisection = n1_bisection.n1
n1_exact, _ = aperture_model.get_aperture_sigmas_at_s(
    s_positions=s_positions,
    method="exact",
    num_rays=32,
)
sigmas_exact = n1_exact.n1

n1_madx_plot = np.where(n1_madx > 9e5, np.inf, n1_madx)

fig, (ax_top, ax_middle, ax_bottom) = plt.subplots(
    3,
    1,
    sharex=True,
    figsize=(10, 9),
    height_ratios=[3, 2, 1],
)

ax_top.plot(s_positions, n1_madx_plot, label="n1 (MAD-X)", lw=1.5)
ax_top.plot(s_positions, sigmas_rays, label="n1 (Xtrack rays)", linestyle="--", lw=1.5)
# ax_top.plot(s_positions, sigmas_bisection, label="n1 (Xtrack bisection)", linestyle=":", lw=1.5)
ax_top.plot(s_positions, sigmas_exact, label="n1 (Xtrack exact)", linestyle="-.", lw=1.5)
ax_top.set_ylabel(r"max beam size [$\sigma$]")
ax_top.set_title(f"IR{which_ir} B1 aperture comparison")
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
    n1_madx_plot - sigmas_exact,
    lw=1.2,
    linestyle="-",
    label=r"$\Delta n_1$ (MAD-X - exact)",
)
ax_bottom.plot(
    s_positions,
    sigmas_rays - sigmas_exact,
    lw=1.2,
    linestyle="--",
    label=r"$\Delta n_1$ (rays - exact)",
)
# ax_bottom.plot(
#     s_positions,
#     sigmas_bisection - sigmas_exact,
#     lw=1.2,
#     linestyle=":",
#     label=r"$\Delta n_1$ (bisection - exact)",
# )
ax_bottom.axhline(0.0, color="k", lw=0.8, linestyle="--")
ax_bottom.set_xlabel("s [m]")
ax_bottom.set_ylabel(r"$\Delta n_1$")
ax_bottom.grid(True)
ax_bottom.legend()

plt.tight_layout()
plt.show()
