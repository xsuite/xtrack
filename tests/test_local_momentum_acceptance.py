# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #
import pytest
import numpy as np
import xtrack as xt


@pytest.fixture(scope="session")
def toy_ring():
    """Build the toy ring once for the entire test session."""
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

    tab = line.get_table()
    needs_aperture = ['Bend', 'Quadrupole']
    aper_size = 0.040  # m

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

    return line


@pytest.fixture(scope="session")
def tt(toy_ring):
    return toy_ring.get_table()


@pytest.fixture(scope="session")
def tw(toy_ring):
    return toy_ring.twiss(method="4d")


@pytest.fixture(scope="session")
def tt_aper(tt):
    return tt.rows[tt.element_type == 'LimitRect']


def test_parameter_validation():
    """
    Validate that invalid input parameters are rejected with clear error messages.
    """
    env = xt.Environment()
    line = env.new_line(components=[env.new('d', xt.Drift, length=1.0)])
    line.set_particle_ref('electron', p0c=1e9)

    cases = [
        (dict(delta_negative_limit=0.0), ValueError, r"delta_negative_limit must be < 0"),
        (dict(delta_positive_limit=0.0), ValueError, r"delta_positive_limit must be > 0"),
        (dict(delta_step_size=0.0),      ValueError, r"delta_step_size must be > 0"),
        (dict(s_start=1.0, s_end=1.0),   ValueError, r"s_start must be < s_end"),
        (dict(n_turns=0),                ValueError, r"n_turns must be > 0"),
    ]

    for kwargs, exc, pattern in cases:
        with pytest.raises(exc, match=pattern):
            line.get_local_momentum_acceptance(
                nemitt_x=1e-6,
                nemitt_y=1e-6,
                method="4d",
                **kwargs,
            )


def test_nemitt_not_provided():
    """
    nemitt_x and nemitt_y are required; omitting either must raise.
    """
    env = xt.Environment()
    line = env.new_line(components=[env.new('d', xt.Drift, length=1.0)])
    line.set_particle_ref('electron', p0c=1e9)

    with pytest.raises(ValueError, match=r"nemitt_x and nemitt_y must be provided"):
        line.get_local_momentum_acceptance(method="4d")


def test_no_particle_ref_raises():
    """
    If the line has no particle_ref set, the method must raise immediately.
    """
    env = xt.Environment()
    line = env.new_line(components=[env.new('d', xt.Drift, length=1.0)])
    # Deliberately no set_particle_ref

    with pytest.raises(ValueError, match=r"Line.particle_ref must be set"):
        line.get_local_momentum_acceptance(
            nemitt_x=1e-6,
            nemitt_y=1e-6,
        )


def test_mutual_exclusivity_errors(toy_ring):
    """
    x/y physical offsets must be mutually exclusive with normalized offsets.
    """
    with pytest.raises(ValueError, match=r"Provide either x_offset or x_norm_offset"):
        toy_ring.get_local_momentum_acceptance(
            nemitt_x=1e-6,
            nemitt_y=1e-6,
            method="4d",
            x_offset=1e-3,
            x_norm_offset=1.0,
        )

    with pytest.raises(ValueError, match=r"Provide either y_offset or y_norm_offset"):
        toy_ring.get_local_momentum_acceptance(
            nemitt_x=1e-6,
            nemitt_y=1e-6,
            method="4d",
            y_offset=1e-3,
            y_norm_offset=1.0,
        )


def test_empty_selection_raises(toy_ring):
    """
    If no elements match the selection filters, the method must raise.
    """
    with pytest.raises(ValueError, match=r"No elements selected for local momentum acceptance computation"):
        toy_ring.get_local_momentum_acceptance(
            nemitt_x=1e-6,
            nemitt_y=1e-6,
            method="4d",
            include_name_pattern="^does_not_exist_.*$",
        )


def test_include_name_pattern_selects_subset(toy_ring, tt, tw):
    """
    A regex name pattern should restrict the output to only the matching elements.
    """
    pattern = "^mqf.*_aper_entry$"
    expected_names = tt.rows[pattern].name

    out = toy_ring.get_local_momentum_acceptance(
        twiss=tw,
        nemitt_x=1e-5,
        nemitt_y=1e-7,
        include_name_pattern=pattern,
        delta_negative_limit=-0.005,
        delta_positive_limit=+0.005,
        delta_step_size=0.001,
        n_turns=512
    )

    assert set(out.name) == set(expected_names)
    assert len(out.name) == len(expected_names)
    assert "s" in out.cols
    assert "deltan" in out.cols
    assert "deltap" in out.cols


def test_s_window_selects_subset(toy_ring, tt, tw):
    """
    s_start / s_end should restrict the output to elements within the window.
    """
    s_start = 0.0
    s_end = tt.s[-1] / 2.0

    tab_in_window = tt.rows[s_start:s_end:'s']
    tab_aper_in_window = tab_in_window.rows[tab_in_window.element_type == 'LimitRect']

    out = toy_ring.get_local_momentum_acceptance(
        twiss=tw,
        nemitt_x=1e-5,
        nemitt_y=1e-7,
        include_type_pattern="LimitRect",
        s_start=s_start,
        s_end=s_end,
        delta_negative_limit=-0.005,
        delta_positive_limit=+0.005,
        delta_step_size=0.001,
        n_turns=512
    )

    assert len(out.s) == len(tab_aper_in_window.name)
    assert set(out.name) == set(tab_aper_in_window.name)


def test_norm_offset_all_survive(toy_ring, tt_aper, tw):
    """
    A small normalized transverse offset should not cause additional losses.
    """
    delta_neg = -0.005
    delta_pos = +0.005
    delta_step = 0.001

    out = toy_ring.get_local_momentum_acceptance(
        twiss=tw,
        include_type_pattern="LimitRect",
        nemitt_x=1e-5,
        nemitt_y=1e-7,
        x_norm_offset=0.1,
        y_norm_offset=0.1,
        delta_negative_limit=delta_neg,
        delta_positive_limit=delta_pos,
        delta_step_size=delta_step,
        n_turns=512
    )

    delta_co = np.array([tw["delta", nn] for nn in tt_aper.name], dtype=float)
    assert np.allclose(out.deltan, delta_co + delta_neg, atol=1e-12)
    assert np.allclose(out.deltap, delta_co + delta_pos, atol=1e-12)


def test_all_survive(toy_ring, tt_aper, tw):
    """
    For a small delta scan all particles survive; deltan/deltap must match
    the scan bounds around the local closed-orbit delta.
    """
    delta_neg = -0.005
    delta_pos = +0.005
    delta_step = 0.001

    out = toy_ring.get_local_momentum_acceptance(
        twiss=tw,
        include_type_pattern="LimitRect",
        nemitt_x=1e-5,
        nemitt_y=1e-7,
        delta_negative_limit=delta_neg,
        delta_positive_limit=delta_pos,
        delta_step_size=delta_step,
        n_turns=512,
        with_progress=False,
        verbose=False,
    )

    assert "s" in out.cols
    assert "deltan" in out.cols
    assert "deltap" in out.cols

    assert len(out.s) == len(tt_aper.name)
    assert np.allclose(out.s, tt_aper.s, atol=1e-12)

    delta_co = np.array([tw["delta", nn] for nn in tt_aper.name], dtype=float)
    assert np.all(out.deltan <= delta_co + 1e-12)
    assert np.all(out.deltap >= delta_co - 1e-12)
    assert np.allclose(out.deltan, delta_co + delta_neg, atol=1e-12)
    assert np.allclose(out.deltap, delta_co + delta_pos, atol=1e-12)


def test_all_lost(toy_ring, tt_aper, tw):
    """
    For a large delta scan all particles are lost; deltan=deltap=0 everywhere.
    """
    out = toy_ring.get_local_momentum_acceptance(
        twiss=tw,
        include_type_pattern="LimitRect",
        nemitt_x=1e-5,
        nemitt_y=1e-7,
        delta_negative_limit=-0.2,
        delta_positive_limit=+0.2,
        delta_step_size=0.2,
        with_progress=False,
        verbose=False,
    )

    assert "s" in out.cols
    assert "deltan" in out.cols
    assert "deltap" in out.cols
    assert len(out.s) == len(tt_aper.name)
    assert np.allclose(out.s, tt_aper.s, atol=1e-12)
    assert np.all(out.deltan <= 0)
    assert np.all(out.deltap >= 0)
    assert np.allclose(out.deltan, 0, atol=1e-12)
    assert np.allclose(out.deltap, 0, atol=1e-12)


@pytest.mark.parametrize(
    "x_offset, y_offset",
    [
        pytest.param(0.050, 0.0, id="lost_by_x_offset"),
        pytest.param(0.0, 0.050, id="lost_by_y_offset"),
    ],
)
def test_all_lost_offset(toy_ring, tt_aper, tw, x_offset, y_offset):
    """
    A transverse offset larger than the aperture causes all particles to be lost.
    """
    out = toy_ring.get_local_momentum_acceptance(
        twiss=tw,
        include_type_pattern="LimitRect",
        nemitt_x=1e-5,
        nemitt_y=1e-7,
        x_offset=x_offset,
        y_offset=y_offset,
        delta_negative_limit=-0.001,
        delta_positive_limit=+0.001,
        delta_step_size=0.0001,
        n_turns=512,
        with_progress=False,
        verbose=False,
    )

    assert "s" in out.cols
    assert "deltan" in out.cols
    assert "deltap" in out.cols
    assert len(out.s) == len(tt_aper.name)
    assert np.allclose(out.s, tt_aper.s, atol=1e-12)
    assert np.all(out.deltan <= 0)
    assert np.all(out.deltap >= 0)
    assert np.allclose(out.deltan, 0.0, atol=1e-12)
    assert np.allclose(out.deltap, 0.0, atol=1e-12)