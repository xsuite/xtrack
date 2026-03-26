import xtrack as xt
from xtrack.aperture import Aperture
from xobjects import ContextCpu
from lhcoptics import LHCOptics
import matplotlib.pyplot as plt
import numpy as np
from xdeps import Table

base = "./acc-models-lhc/" # "https://cern.ch/acc-models/lhc/hl19/"
lhc = xt.load(f"{base}/xsuite/lhc_aperture.json")
lhc.vars.load(f"{base}/strengths/cycle_round_v0/opt_6000.madx")
lhc.set_particle_ref(p0c=450e9)

lhc.vars.update({
    'on_sep1v':  2,
    'on_x1hs': 295,
    'on_sep5h':  2,
    'on_x5vs': 295,
    'on_sep2h': -3.5,
    'on_x2v':  170,
    'on_a2h': 40,
    'on_alice': 7000/450,
    'on_sep8v': -3.5,
    'on_x8h': -170,
    'on_a8v': -40,
    'on_lhcb': 7000/450,
})

opt = LHCOptics.from_xsuite(lhc)

mad = opt.make_madx_model()
apm = mad.get_ap_irs()

# load in metadata
lhc.b1.metadata["aperture_offsets"] = {}
lhc.b2.metadata["aperture_offsets"] = {}
for ipn in range(1, 9):
    for beam in "14":
        tfs = Table.from_tfs(f"./temp/offset.ip{ipn}.b{beam}.tfs")
        line = lhc.b1 if beam == "1" else lhc.b2
        line.metadata["aperture_offsets"][f"ip{ipn}"] = tfs._data.copy()

context = ContextCpu(omp_num_threads='auto')

lines = {
    "b1": lhc.b1,
    #"b2": lhc.b2,
}
apertures = {
    beam: Aperture.from_line_with_madx_metadata(
        line,
        num_profile_points=100,
        include_offsets=True,
        context=context,
    )
    for beam, line in lines.items()
}
line_tables = {beam: line.get_table() for beam, line in lines.items()}


def _get_nearest_element_name(line_table, s_position):
    idx = np.searchsorted(np.asarray(line_table.s, dtype=float), s_position, side="right") - 1
    idx = int(np.clip(idx, 0, len(line_table.name) - 1))
    return line_table.name[idx]


for ir_name in sorted(apm):
    ir = apm[ir_name]
    beam = ir_name[-2:]

    if beam == 'b2':
        continue

    line = lines[beam]
    line_table = line_tables[beam]
    aperture = apertures[beam]

    aperture.halo_params.update(
        {
            "emitx_norm": ir.exn,
            "emity_norm": ir.eyn,
            "delta_rms": ir.dp_bucket_size,
            "tol_co": ir.co_radius,
            "tol_disp": ir.paras_dx,  # MAD-X has different settings for x/y
            "tol_disp_ref_dx": ir.dqf,
            "tol_disp_ref_beta": ir.betaqfx,
            "tol_energy": 0.0,  # TO CHECK
            "tol_beta_beating": ir.beta_beating,  # MAD-X has different settings for x/y
        }
    )

    ip_name = f"ip{ir_name[2]}"
    s_ip_m = ir.rows[f"{ip_name}.*"].s[0]
    s_ip_x = line_table.rows[f"{ip_name}.*"].s[0]
    s_local = np.asarray(ir.s - s_ip_m, dtype=float)
    s_positions = np.mod(s_local + s_ip_x, line.get_length())
    order = np.argsort(s_positions)
    undo_order = np.empty_like(order)
    undo_order[order] = np.arange(len(order))

    n1_table, _ = aperture.get_aperture_sigmas_at_s(
        s_positions=s_positions[order],
        method="rays",
    )
    sigmas = np.asarray(n1_table.n1[undo_order], dtype=float)
    s_positions_ring = np.asarray(n1_table.s[undo_order], dtype=float)

    min_idx = int(np.argmin(sigmas))
    min_n1 = float(sigmas[min_idx])
    min_s = float(s_positions_ring[min_idx])
    element_name = (
        ir.name[min_idx]
        if hasattr(ir, "name") and len(ir.name) == len(sigmas)
        else _get_nearest_element_name(line_table, min_s)
    )
    print(f"{ir_name}: min n1={min_n1:.3f} at {element_name} (s={min_s:.6f} m)")

    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(
        s_local, np.where(ir.n1 > 9e5, np.inf, ir.n1), label='n1 (MAD-X)'
    )
    axs[0].plot(s_local, sigmas, label='n1 (Xtrack)')
    axs[0].set_ylabel(r'max beam size [$\sigma$]')
    axs[0].legend()

    aperture.plot_extents(
        s_positions=s_positions,
        sigmas=min_n1,
        plot_s_positions=s_local,
        axs=axs[1:],
    )
    axs[1].set_title(ir_name)
    axs[2].set_xlabel(f's - {ip_name} [m]')

    fig.tight_layout()
    plt.show()
