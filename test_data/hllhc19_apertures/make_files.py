import xtrack as xt
from xtrack.aperture import Aperture
from xobjects import ContextCpu
from lhcoptics import LHCOptics
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from xdeps import Table
from xobjects.general import allclose_with_outliers

base_dir = Path(__file__).resolve().parent
acc_models_dir = base_dir / "acc-models-lhc"
plot = False

lhc = xt.load(acc_models_dir / "xsuite" / "lhc_aperture.json")
lhc.vars.load(acc_models_dir / "strengths" / "cycle_round_v0" / "opt_6000.madx")
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
        tfs = Table.from_tfs(base_dir / "temp" / f"offset.ip{ipn}.b{beam}.tfs")
        line = lhc.b1 if beam == "1" else lhc.b2
        tfs_data = tfs._data.copy()
        tfs_data['name'] = [name for name in tfs_data['name']]  # clear StringArray
        line.metadata["aperture_offsets"][f"ip{ipn}"] = tfs_data

lhc.to_json(base_dir / "lhc_aperture.json")

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

apertures['b1'].to_json(base_dir / "aperture_model_b1.json")
apertures['b1'] = Aperture.from_json(base_dir / "aperture_model_b1.json", lhc.b1)

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
    reference_halo_params = {
        key: float(aperture.halo_params[key])
        for key in (
            "emitx_norm",
            "emity_norm",
            "delta_rms",
            "tol_co",
            "tol_disp",
            "tol_disp_ref_dx",
            "tol_disp_ref_beta",
            "tol_energy",
            "tol_beta_beating",
            "halo_x",
            "halo_y",
            "halo_r",
            "halo_primary",
        )
    }

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

    # Clean up MAD-X data
    n1_madx = np.where(ir.n1 >= 7000, np.nan, ir.n1)  # put extreme values to nan
    valid_mask = np.isfinite(n1_madx)
    non_cont_mask = valid_mask & np.r_[True, ~valid_mask[:-1]] & np.r_[~valid_mask[1:], True]
    n1_madx[non_cont_mask] = np.nan  # put non continuous values to nan

    n1_xt = sigmas

    absdiffs = np.abs(n1_madx - n1_xt)
    reldiffs = np.abs((n1_madx - n1_xt) / n1_madx)
    mask = (reldiffs > 0.01) & np.isfinite(reldiffs)
    lines_where = s_local[mask]
    no_outliers = len(lines_where)
    print(f"==> Outliers: {no_outliers}")


    print(f"MAD-X vs Xtrack: abs diff = {np.nanmax(absdiffs)}, rel diff = {np.nanmax(reldiffs)}")
    sorted_abs, sorted_rel = np.sort(absdiffs[np.isfinite(absdiffs)]), np.sort(reldiffs[np.isfinite(reldiffs)])
    if no_outliers > 0:
        print(f"MAD-X vs Xtrack: abs mean diff = {np.nanmax(sorted_abs[:-no_outliers])}, rel mean diff = {np.nanmax(sorted_rel[:-no_outliers])}")
    else:
        print("No outliers")

    # 99% of points within 1% tolerance
    assert allclose_with_outliers(n1_madx[np.isfinite(n1_madx)], n1_xt[np.isfinite(n1_madx)], rtol=0.01, max_outliers=int(len(n1_madx) / 100))

    with open(base_dir / f"{ir_name}.json", "w") as fid:
        json.dump(
            {
                "beam": beam,
                "ip_name": ip_name,
                "s_local": np.asarray(s_local, dtype=float).tolist(),
                "n1_madx": np.asarray(n1_madx, dtype=float).tolist(),
                "halo_params": reference_halo_params,
            },
            fid,
            indent=2,
        )

    if plot:
        plot_order = np.argsort(s_local)
        s_plot = s_local[plot_order]
        n1_madx_plot = n1_madx[plot_order]
        n1_xt_plot = n1_xt[plot_order]
        lines_where_plot = np.sort(lines_where)

        plt.figure()
        plt.plot(s_plot, n1_madx_plot, label='n1 (MAD-X)')
        plt.plot(s_plot, n1_xt_plot, label='n1 (Xtrack)')
        [plt.axvline(l, linestyle='--', c='k') for l in lines_where_plot]

        plt.xlabel(f's - {ip_name} [m]')
        plt.ylabel(r'max beam size [$\sigma$]')
        plt.legend()
        plt.show()
