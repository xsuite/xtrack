import matplotlib.pyplot as plt
import numpy as np

import xtrack as xt
from xdeps import Table
from xobjects import ContextCpu
from xtrack.aperture import Aperture


BASE = "./acc-models-lhc/"
BEAM = "b2"
ELEMENT_NAME = "bpmwi.4r8.b2"
WINDOW = 10.0
SURVEY_RESOLUTION = 1.0


def _poly2d_to_hom(points):
    return np.vstack([
        points[:, 0],
        points[:, 1],
        np.zeros(len(points)),
        np.ones(len(points)),
    ])


def _load_line():
    lhc = xt.load(f"{BASE}/xsuite/lhc_aperture.json")
    lhc.vars.load(f"{BASE}/strengths/cycle_round_v0/opt_6000.madx")
    lhc.set_particle_ref(p0c=450e9)

    line = lhc.b2 if BEAM == "b2" else lhc.b1
    line.metadata["aperture_offsets"] = {}
    for ipn in range(1, 9):
        tfs = Table.from_tfs(f"./temp/offset.ip{ipn}.{BEAM}.tfs")
        tfs_data = tfs._data.copy()
        tfs_data["reversed"] = BEAM == "b2"
        line.metadata["aperture_offsets"][f"ip{ipn}"] = tfs_data

    return line


def main():
    line = _load_line()
    aperture = Aperture.from_line_with_madx_metadata(
        line,
        num_profile_points=100,
        include_offsets=True,
        context=ContextCpu(omp_num_threads="auto"),
        _skip_validity_check=True,
    )

    survey = line.survey()
    element_s = float(line.get_s_position(ELEMENT_NAME))

    bounds_table = aperture.get_bounds_table()
    bound_mask = np.asarray(bounds_table.pipe_name) == ELEMENT_NAME
    profile_s = np.asarray(bounds_table.s[bound_mask], dtype=float)
    profile_s_start = np.asarray(bounds_table.s_start[bound_mask], dtype=float)
    profile_s_end = np.asarray(bounds_table.s_end[bound_mask], dtype=float)

    survey_window_s = np.arange(
        element_s - WINDOW,
        element_s + WINDOW + SURVEY_RESOLUTION,
        SURVEY_RESOLUTION,
    )
    survey_cut_s = np.unique(np.concatenate([
        [element_s - WINDOW, element_s + WINDOW],
        survey_window_s,
        profile_s_start,
        profile_s,
        profile_s_end,
    ]))
    line_sliced = line.copy()
    line_sliced.cut_at_s(survey_cut_s)
    survey_sliced = line_sliced.survey()
    survey_s = np.asarray(survey_sliced.s, dtype=float)
    survey_mask = (element_s - WINDOW <= survey_s) & (survey_s <= element_s + WINDOW)

    # cross_sections_at_s returns both the local polygon and the pose that moves
    # the polygon into the survey/world frame, as used in the other examples.
    sections_table = aperture.cross_sections_at_s(profile_s)
    bound_points_table = aperture.poses_at_s(survey_cut_s)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        np.asarray(survey_sliced.Z, dtype=float)[survey_mask],
        np.asarray(survey_sliced.X, dtype=float)[survey_mask],
        ".-",
        color="k",
        label="survey",
    )
    for label, s_values, color, marker in [
        ("s_start", profile_s_start, "tab:blue", "o"),
        ("s", profile_s, "tab:red", "x"),
        ("s_end", profile_s_end, "tab:green", "s"),
    ]:
        poses = aperture.poses_at_s(s_values)
        ax.scatter(
            poses[:, 2, 3],
            poses[:, 0, 3],
            color=color,
            marker=marker,
            s=60,
            label=label,
            zorder=5,
        )

    for idx, (section, pose, s_position) in enumerate(
        zip(sections_table.cross_section, sections_table.pose, profile_s)
    ):
        section_in_world = pose @ _poly2d_to_hom(section)
        x = section_in_world[0]
        z = section_in_world[2]
        ax.plot(z, x, "-", lw=1.5, label=f"profile {idx}, s={s_position:.6f}")

    ax.set_title(f"{ELEMENT_NAME}: survey and installed profile in Z-X")
    ax.set_xlabel("Z [m]")
    ax.set_ylabel("X [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
