import xtrack as xt
from xtrack.aperture import Aperture
from xobjects import ContextCpu
from lhcoptics import LHCOptics
import matplotlib.pyplot as plt
import numpy as np
from xdeps import Table
import warnings

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
    for beam in "12":
        tfs = Table.from_tfs(f"./temp/offset.ip{ipn}.b{beam}.tfs")
        line = lhc.b1 if beam == "1" else lhc.b2
        tfs_data = tfs._data.copy()
        tfs_data["reversed"] = beam == "2"
        line.metadata["aperture_offsets"][f"ip{ipn}"] = tfs_data

context = ContextCpu(omp_num_threads='auto')

lines = {
    "b1": lhc.b1,
    "b2": lhc.b2,
}
apertures = {
    beam: Aperture.from_line_with_madx_metadata(
        line,
        num_profile_points=100,
        include_offsets=True,
        context=context,
        _skip_validity_check=True,
    )
    for beam, line in lines.items()
}
line_tables = {beam: line.get_table() for beam, line in lines.items()}


def _get_nearest_element_name(line_table, s_position):
    idx = np.searchsorted(np.asarray(line_table.s, dtype=float), s_position, side="right") - 1
    idx = int(np.clip(idx, 0, len(line_table.name) - 1))
    return line_table.name[idx]


def _offset_s_positions(survey_table, offset_data):
    s_by_name = {}
    s_values = np.asarray(survey_table.s, dtype=float)
    survey_names = np.asarray(survey_table.name)
    for idx, survey_name in enumerate(survey_names):
        mad_name = Aperture._guess_original_mad_name(survey_name)
        s_by_name.setdefault(mad_name, (idx, float(s_values[idx])))

    s_positions = []
    s_end_positions = []
    missing = []
    keep_indices = []
    for name in offset_data["name"]:
        if name in s_by_name:
            idx, s0 = s_by_name[name]
            s1 = float(s_values[idx + 1]) if idx + 1 < len(s_values) else s0
            s_positions.append(s0)
            s_end_positions.append(s1)
            keep_indices.append(True)
        else:
            missing.append(name)
            keep_indices.append(False)

    if missing:
        warnings.warn(
            "Skipping offset rows not present in the model: "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else ""),
            stacklevel=2,
        )

    keep_indices = np.asarray(keep_indices, dtype=bool)
    filtered_offset_data = {}
    for key, value in offset_data.items():
        value_arr = np.asarray(value)
        if value_arr.ndim > 0 and len(value_arr) == len(keep_indices):
            filtered_offset_data[key] = value_arr[keep_indices]
        else:
            filtered_offset_data[key] = value
    reference_name = offset_data["reference"]
    reference_entry = s_by_name.get(reference_name, None)
    reference_s = None if reference_entry is None else reference_entry[1]
    if reference_s is None:
        warnings.warn(
            f"Skipping unresolved offset reference {reference_name!r}",
            stacklevel=2,
        )

    return (
        np.asarray(s_positions, dtype=float),
        np.asarray(s_end_positions, dtype=float),
        filtered_offset_data,
        reference_s,
    )


def _wrap_local_s(s, line_length):
    half_length = 0.5 * line_length
    return np.where(s > half_length, s - line_length, np.where(s < -half_length, s + line_length, s))


def _as_scalar(value):
    arr = np.asarray(value, dtype=float)
    return float(arr.reshape(-1)[0])


for ir_name in sorted(apm):
    ir = apm[ir_name]
    beam = ir_name[-2:]

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
            "tol_disp_ref": ir.dqf,
            "tol_disp_ref_beta": ir.betaqfx,
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
    offset_data = line.metadata["aperture_offsets"][ip_name]
    line_length = line.get_length()
    offset_s_start, offset_s_end, offset_data, offset_reference_s = _offset_s_positions(
        line.survey(), offset_data
    )
    offset_s_start = _wrap_local_s(offset_s_start - s_ip_x, line_length)
    offset_s_end = _wrap_local_s(offset_s_end - s_ip_x, line_length)
    offset_reference_s = (
        None
        if offset_reference_s is None
        else _wrap_local_s(offset_reference_s - s_ip_x, line_length)
    )
    if offset_reference_s is not None:
        offset_reference_s = _as_scalar(offset_reference_s)
    ir_s_min = float(np.min(s_local))
    ir_s_max = float(np.max(s_local))
    in_ir = (offset_s_end >= ir_s_min) & (offset_s_start <= ir_s_max)
    offset_s_start = offset_s_start[in_ir]
    offset_s_end = offset_s_end[in_ir]
    filtered_offset_data = {}
    for key, value in offset_data.items():
        value_arr = np.asarray(value)
        if value_arr.ndim > 0 and len(value_arr) == len(in_ir):
            filtered_offset_data[key] = value_arr[in_ir]
        else:
            filtered_offset_data[key] = value
    offset_data = filtered_offset_data

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

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
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
        axs=axs[1:3],
    )
    axs[1].set_title(ir_name)
    axs[2].set_xlabel(f's - {ip_name} [m]')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for label, values, color in [
        ("x", offset_data["x_off"], colors[0]),
        ("dx", offset_data["dx_off"], colors[1]),
        ("ddx", offset_data["ddx_off"], colors[2]),
        ("y", offset_data["y_off"], colors[3]),
        ("dy", offset_data["dy_off"], colors[4]),
        ("ddy", offset_data["ddy_off"], colors[5]),
    ]:
        first = True
        for s0, s1, value in zip(offset_s_start, offset_s_end, values):
            axs[3].plot(
                [s0, s1],
                [value, value],
                linestyle="-",
                marker=None,
                lw=1.2,
                alpha=0.8,
                label=label if first else None,
                color=color,
            )
            first = False
    if offset_reference_s is not None:
        axs[3].axvline(offset_reference_s, color="k", linestyle="--", lw=1.0, label="reference")
    axs[3].set_ylabel("offset data")
    axs[3].set_xlabel(f"s - {ip_name} [m]")
    axs[3].grid(True)
    axs[3].legend(ncol=3, fontsize="small")

    fig.tight_layout()
    plt.show()
