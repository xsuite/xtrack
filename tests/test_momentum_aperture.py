# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #
import pytest
import numpy as np
import xtrack as xt


def build_toy_ring_with_apertures():
    lbend = 3
    angle = np.pi / 2

    env = xt.Environment()

    line = env.new_line(components=[
        env.new('mqf.1', xt.Quadrupole, length=0.3, k1=0.1),
        env.new('d1.1',  xt.Drift, length=1),
        env.new('mb1.1', xt.Bend, length=lbend, angle=angle),
        env.new('d2.1',  xt.Drift, length=1),

        env.new('mqd.1', xt.Quadrupole, length=0.3, k1=-0.7),
        env.new('d3.1',  xt.Drift, length=1),
        env.new('mb2.1', xt.Bend, length=lbend, angle=angle),
        env.new('d4.1',  xt.Drift, length=1),

        env.new('mqf.2', xt.Quadrupole, length=0.3, k1=0.1),
        env.new('d1.2',  xt.Drift, length=1),
        env.new('mb1.2', xt.Bend, length=lbend, angle=angle),
        env.new('d2.2',  xt.Drift, length=1),

        env.new('mqd.2', xt.Quadrupole, length=0.3, k1=-0.7),
        env.new('d3.2',  xt.Drift, length=1),
        env.new('mb2.2', xt.Bend, length=lbend, angle=angle),
        env.new('d4.2',  xt.Drift, length=1),
    ])

    line.set_particle_ref('electron', p0c=1e9)
    line.configure_bend_model(core='full', edge=None)

    # Insert apertures
    tab = line.get_table()
    needs_aperture = ['Bend', 'Quadrupole']
    aper_size = 0.040 # m

    placements = []
    for nn, ee in zip(tab.name, tab.element_type):
        if ee not in needs_aperture:
            continue

        env.new(
            f'{nn}_aper_entry', xt.LimitRect,
            min_x=-aper_size, max_x=aper_size,
            min_y=-aper_size, max_y=aper_size
        )
        placements.append(env.place(f'{nn}_aper_entry', at=f'{nn}@start'))

        env.new(
            f'{nn}_aper_exit', xt.LimitRect,
            min_x=-aper_size, max_x=aper_size,
            min_y=-aper_size, max_y=aper_size
        )
        placements.append(env.place(f'{nn}_aper_exit', at=f'{nn}@end'))

    line.insert(placements)

    return env, line


def test_parameter_validation():
    """
    Validate that invalid input parameters are rejected with clear error messages.
    """
    _, line = build_toy_ring_with_apertures()

    cases = [
        # (kwargs, expected_exc, regex)
        (dict(delta_negative_limit=0.0), ValueError, r"delta_negative_limit must be < 0"),
        (dict(delta_positive_limit=0.0), ValueError, r"delta_positive_limit must be > 0"),
        (dict(delta_step_size=0.0),      ValueError, r"delta_step_size must be > 0"),
        (dict(s_start=1.0, s_end=1.0),   ValueError, r"s_start must be < s_end"),
        (dict(n_turns=0),                ValueError, r"n_turns must be > 0"),
        (dict(skip_elements=-1),         ValueError, r"skip_elements must be >= 0"),
        (dict(process_elements=0),       ValueError, r"process_elements must be > 0"),
    ]

    for kwargs, exc, pattern in cases:
        with pytest.raises(exc, match=pattern):
            line.momentum_aperture(
                nemitt_x=1e-6,
                nemitt_y=1e-6,
                method="4d",
                **kwargs,
            )


def test_forbid_resonance_crossing_not_implemented():
    """
    forbid_resonance_crossing is currently a reserved option and must raise.
    """
    _, line = build_toy_ring_with_apertures()

    with pytest.raises(NotImplementedError, match=r"forbid_resonance_crossing is not implemented"):
        line.momentum_aperture(
            nemitt_x=1e-6,
            nemitt_y=1e-6,
            method="4d",
            forbid_resonance_crossing=1,
        )


def test_mutual_exclusivity_errors():
    """
    x/y physical offsets must be mutually exclusive with normalized offsets.
    """
    _, line = build_toy_ring_with_apertures()

    # x-plane conflict
    with pytest.raises(ValueError, match=r"Provide either x_offset or x_norm_offset"):
        line.momentum_aperture(
            nemitt_x=1e-6,
            nemitt_y=1e-6,
            method="4d",
            x_offset=1e-3,
            x_norm_offset=1.0,
        )

    # y-plane conflict
    with pytest.raises(ValueError, match=r"Provide either y_offset or y_norm_offset"):
        line.momentum_aperture(
            nemitt_x=1e-6,
            nemitt_y=1e-6,
            method="4d",
            y_offset=1e-3,
            y_norm_offset=1.0,
        )


def test_empty_selection_raises():
    """
    If no elements match the selection filters, the method must raise.
    """
    _, line = build_toy_ring_with_apertures()

    with pytest.raises(ValueError, match=r"No elements selected for momentum aperture computation"):
        line.momentum_aperture(
            nemitt_x=1e-6,
            nemitt_y=1e-6,
            method="4d",
            include_name_pattern="does_not_exist_*",
        )


def test_all_survive():
    """
    For a small delta scan and relatively short tracking, particles must survive in this toy
    ring, so deltan/deltap should match the scan bounds around the local closed-orbit delta.
    """
    _, line = build_toy_ring_with_apertures()

    tab = line.get_table()
    tab_aper = tab.rows[tab.element_type == 'LimitRect']
    n_aper = len(tab_aper.name)

    tw = line.twiss(method="4d")

    delta_neg = -0.005
    delta_pos = +0.005
    delta_step = 0.001

    out = line.momentum_aperture(
        twiss=tw,
        include_type_pattern="LimitRect",  # avoid duplicates at the same s
        nemitt_x=1e-5,
        nemitt_y=1e-7,
        delta_negative_limit=delta_neg,
        delta_positive_limit=delta_pos,
        delta_step_size=delta_step,
        n_turns=512,
        method="4d",
        with_progress=False,
        verbose=False,
    )

    assert "s" in out.cols
    assert "deltan" in out.cols
    assert "deltap" in out.cols

    assert len(out.s) == n_aper
    assert np.allclose(out.s, tab_aper.s, atol=1e-12)

    # expected bounds include the local closed-orbit delta at each scanned element
    delta_co = np.array([tw["delta", nn] for nn in tab_aper.name], dtype=float)
    expected_deltan = delta_co + delta_neg
    expected_deltap = delta_co + delta_pos

    assert np.all(out.deltan <= delta_co + 1e-12)
    assert np.all(out.deltap >= delta_co - 1e-12)

    assert np.allclose(out.deltan, expected_deltan, atol=1e-12)
    assert np.allclose(out.deltap, expected_deltap, atol=1e-12)


def test_all_lost():
    """
    For a large delta scan exceeding the momentum acceptance of the toy ring,
    all particles are expected to be lost at the rectangular apertures.
    Therefore, no surviving particles should be found and the returned
    momentum aperture must be deltan=deltap=0 at all locations.
    """
    _, line = build_toy_ring_with_apertures()

    tab = line.get_table()
    n_aper = len(tab.rows[tab.element_type == 'LimitRect'].name)

    tw = line.twiss(method="4d")

    delta_neg = -0.2
    delta_pos = +0.2
    delta_step = 0.2

    out = line.momentum_aperture(
        twiss=tw,
        include_type_pattern="LimitRect", # avoid duplicates at the same s
        nemitt_x=1e-5,
        nemitt_y=1e-7,
        delta_negative_limit=delta_neg,
        delta_positive_limit=delta_pos,
        delta_step_size=delta_step,
        n_turns=512,
        method="4d",
        with_progress=False,
        verbose=False,
    )

    assert "s" in out.cols
    assert "deltan" in out.cols
    assert np.all(out.deltan <= 0)
    assert "deltap" in out.cols
    assert np.all(out.deltap >= 0)

    assert len(out.s) == n_aper

    assert np.allclose(out.s, tab.rows[tab.element_type == 'LimitRect'].s, atol=1e-12)
    assert np.allclose(out.deltan, 0, atol=1e-12)
    assert np.allclose(out.deltap, 0, atol=1e-12)


@pytest.mark.parametrize(
    "x_offset, y_offset",
    [
        pytest.param(0.050, 0.0, id="lost_by_x_offset"),
        pytest.param(0.0, 0.050, id="lost_by_y_offset"),
    ],
)
def test_all_lost_offset(x_offset, y_offset):
    """
    For a tiny delta scan, but with a transverse offset larger than the rectangular
    half-aperture, all particles must be lost, therefore deltan=deltap=0 everywhere.
    """
    _, line = build_toy_ring_with_apertures()

    tab = line.get_table()
    tab_aper = tab.rows[tab.element_type == 'LimitRect']
    n_aper = len(tab_aper.name)

    tw = line.twiss(method="4d")

    delta_neg = -0.001
    delta_pos = +0.001
    delta_step = 0.0001

    out = line.momentum_aperture(
        twiss=tw,
        include_type_pattern="LimitRect",
        nemitt_x=1e-5,
        nemitt_y=1e-7,
        x_offset=x_offset,
        y_offset=y_offset,
        delta_negative_limit=delta_neg,
        delta_positive_limit=delta_pos,
        delta_step_size=delta_step,
        n_turns=512,
        method="4d",
        with_progress=False,
        verbose=False,
    )

    assert "s" in out.cols
    assert "deltan" in out.cols
    assert np.all(out.deltan <= 0)
    assert "deltap" in out.cols
    assert np.all(out.deltap >= 0)

    assert len(out.s) == n_aper

    assert np.allclose(out.s, tab_aper.s, atol=1e-12)
    assert np.allclose(out.deltan, 0.0, atol=1e-12)
    assert np.allclose(out.deltap, 0.0, atol=1e-12)