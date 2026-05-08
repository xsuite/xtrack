import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from xdeps import Table
from xobjects.general import allclose_with_outliers
from xobjects import ContextCpu

from xtrack.aperture import Aperture


BASE_DIR = Path(__file__).resolve().parents[3] / "test_data" / "hllhc19_apertures"
OFFSET_COLUMNS = [
    ("x", "x_off"),
    ("dx", "dx_off"),
    ("ddx", "ddx_off"),
    ("y", "y_off"),
    ("dy", "dy_off"),
    ("ddy", "ddy_off"),
]


def _load_reference_irs():
    refs = {}
    for path in sorted(BASE_DIR.glob("ir*b*.json")):
        refs[path.stem] = json.loads(path.read_text())
    return refs


def _get_nearest_element_name(line_table, s_position):
    idx = np.searchsorted(np.asarray(line_table.s, dtype=float), s_position, side="right") - 1
    idx = int(np.clip(idx, 0, len(line_table.name) - 1))
    return line_table.name[idx]


def _offset_s_positions(survey_table, offset_data):
    """Resolve MAD-X offset row names to Xtrack survey entry/exit positions."""
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


def _filter_offset_data(offset_data, mask):
    filtered_offset_data = {}
    for key, value in offset_data.items():
        value_arr = np.asarray(value)
        if value_arr.ndim > 0 and len(value_arr) == len(mask):
            filtered_offset_data[key] = value_arr[mask]
        else:
            filtered_offset_data[key] = value
    return filtered_offset_data


def _wrap_local_s(s, line_length):
    half_length = 0.5 * line_length
    return np.where(
        s > half_length,
        s - line_length,
        np.where(s < -half_length, s + line_length, s),
    )


def _beam_s_direction(beam):
    return -1.0 if beam == "b2" else 1.0


def _ir_local_to_ring_s(s_local, s_ip, line_length, beam):
    return np.mod(s_ip + _beam_s_direction(beam) * s_local, line_length)


def _ring_to_ir_local_s(s_ring, s_ip, line_length, beam):
    return _beam_s_direction(beam) * _wrap_local_s(s_ring - s_ip, line_length)


def _as_scalar(value):
    arr = np.asarray(value, dtype=float)
    return float(arr.reshape(-1)[0])


def _load_lines():
    lhc = xt.load(BASE_DIR / "lhc_aperture.json")

    lines = {
        "b1": lhc.b1,
        "b2": lhc.b2,
    }

    return lines


def main():
    references = _load_reference_irs()
    lines = _load_lines()
    context = ContextCpu(omp_num_threads="auto")

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

    for ir_name in sorted(references):
        reference = references[ir_name]
        beam = reference["beam"]
        ip_name = reference["ip_name"]

        line = lines[beam]
        line_table = line_tables[beam]
        aperture = apertures[beam]

        aperture.halo_params.update(reference["halo_params"])

        s_ip_x = line_table.rows[f"{ip_name}.*"].s[0]
        s_positions = np.asarray(reference["s_positions"], dtype=float)
        s_local = _ring_to_ir_local_s(s_positions, s_ip_x, line.get_length(), beam)
        order = np.argsort(s_positions)
        undo_order = np.empty_like(order)
        undo_order[order] = np.arange(len(order))

        n1_table, _ = aperture.get_aperture_sigmas_at_s(
            s_positions=s_positions[order],
            method="rays",
        )
        sigmas = np.asarray(n1_table.n1[undo_order], dtype=float)
        s_positions_ring = np.asarray(n1_table.s[undo_order], dtype=float)
        n1_madx = np.asarray(reference["n1_madx"], dtype=float)

        min_idx = int(np.argmin(sigmas))
        min_n1 = float(sigmas[min_idx])
        min_s = float(s_positions_ring[min_idx])
        element_name = _get_nearest_element_name(line_table, min_s)
        print(f"{ir_name}: min n1={min_n1:.3f} at {element_name} (s={min_s:.6f} m)")

        n1_madx = np.where(n1_madx >= 7000, np.nan, n1_madx)
        valid_mask = np.isfinite(n1_madx)
        non_cont_mask = valid_mask & np.r_[True, ~valid_mask[:-1]] & np.r_[~valid_mask[1:], True]
        n1_madx[non_cont_mask] = np.nan

        absdiffs = np.abs(n1_madx - sigmas)
        reldiffs = np.abs((n1_madx - sigmas) / n1_madx)
        mask = (reldiffs > 0.01) & np.isfinite(reldiffs)
        lines_where = s_local[mask]
        no_outliers = len(lines_where)
        print(f"==> Outliers: {no_outliers}")
        print(f"MAD-X vs Xtrack: abs diff = {np.nanmax(absdiffs)}, rel diff = {np.nanmax(reldiffs)}")
        sorted_abs, sorted_rel = np.sort(absdiffs[np.isfinite(absdiffs)]), np.sort(reldiffs[np.isfinite(reldiffs)])
        if no_outliers > 0:
            print(
                f"MAD-X vs Xtrack: abs mean diff = {np.nanmax(sorted_abs[:-no_outliers])}, "
                f"rel mean diff = {np.nanmax(sorted_rel[:-no_outliers])}"
            )
        else:
            print("No outliers")

        # 99% of points within 1% tolerance
        finite_mask = np.isfinite(n1_madx)
        assert allclose_with_outliers(
            n1_madx[finite_mask],
            sigmas[finite_mask],
            rtol=0.01,
            max_outliers=int(len(n1_madx) / 100),
        )

        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 10))

        # ---------------------------------------------------------------------
        # n1 comparison
        # ---------------------------------------------------------------------
        axs[0].plot(s_local, n1_madx, label="n1 (MAD-X)")
        axs[0].plot(s_local, sigmas, label="n1 (Xtrack)", linestyle="--")
        axs[0].set_ylabel(r"max beam size [$\sigma$]")
        axs[0].legend()

        aperture.plot_extents(
            s_positions=s_positions,
            sigmas=min_n1,
            plot_s_positions=s_local,
            axs=axs[1:3],
        )
        axs[1].set_title(ir_name)
        axs[2].set_xlabel(f"s - {ip_name} [m]")

        # ---------------------------------------------------------------------
        # Offset diagnostics
        # ---------------------------------------------------------------------
        offset_data = line.metadata["aperture_offsets"][ip_name]
        line_length = line.get_length()
        offset_s_start, offset_s_end, offset_data, offset_reference_s = _offset_s_positions(
            line.survey(), offset_data
        )
        offset_s_start = _ring_to_ir_local_s(offset_s_start, s_ip_x, line_length, beam)
        offset_s_end = _ring_to_ir_local_s(offset_s_end, s_ip_x, line_length, beam)
        offset_reference_s = _as_scalar(_ring_to_ir_local_s(offset_reference_s, s_ip_x, line_length, beam))

        offset_s_min = np.minimum(offset_s_start, offset_s_end)
        offset_s_max = np.maximum(offset_s_start, offset_s_end)
        ir_s_min = float(np.min(s_local))
        ir_s_max = float(np.max(s_local))
        in_ir = (offset_s_max >= ir_s_min) & (offset_s_min <= ir_s_max)
        offset_s_start = offset_s_min[in_ir]
        offset_s_end = offset_s_max[in_ir]
        offset_data = _filter_offset_data(offset_data, in_ir)

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for color_idx, (label, column) in enumerate(OFFSET_COLUMNS):
            first = True
            for s0, s1, value in zip(offset_s_start, offset_s_end, offset_data[column]):
                axs[3].plot(
                    [s0, s1],
                    [value, value],
                    linestyle="-",
                    marker=None,
                    lw=1.2,
                    alpha=0.8,
                    label=label if first else None,
                    color=colors[color_idx],
                )
                first = False
        axs[3].axvline(
            offset_reference_s,
            color="k",
            linestyle="--",
            lw=1.0,
            label="reference",
        )
        axs[3].set_ylabel("offset data")
        axs[3].set_xlabel(f"s - {ip_name} [m]")
        axs[3].grid(True)
        axs[3].legend(ncol=3, fontsize="small")

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
