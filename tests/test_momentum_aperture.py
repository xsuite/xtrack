# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #
import xtrack as xt
import numpy as np
import pytest
import re

def build_toy_ring_with_apertures(aper_size):
    # Toy ring parameters
    p0c = 1.2e9
    pdg_id = 11
    mass0 = xt.ELECTRON_MASS_EV
    lbend = 0.5
    theta = np.pi / 2
    k0_bend = theta / 6 / lbend
    h_bend  = theta / 6 / lbend

    # Helpers
    def fodo_arc_block(block_id: int):
        el = []
        # QF
        el.append(env.new(f"mqf.{2*block_id-1}", xt.Quadrupole, length=0.3, k1=0.1))
        # Drift
        el.append(env.new(f"d{block_id}.1",  xt.Drift, length=0.5))
        el.append(env.new(f"d{block_id}.2",  xt.Drift, length=0.5))
        # Bend
        el.append(env.new(f"mb1.{block_id}.0", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb1.{block_id}.1", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb1.{block_id}.2", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb1.{block_id}.3", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb1.{block_id}.4", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb1.{block_id}.5", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        # Drift
        el.append(env.new(f"d{block_id}.3",  xt.Drift, length=0.5))
        el.append(env.new(f"d{block_id}.4",  xt.Drift, length=0.5))
        # QD
        el.append(env.new(f"mqd.{2*block_id-1}", xt.Quadrupole, length=0.3, k1=-0.7))
        # Drift
        el.append(env.new(f"d{block_id}.5",  xt.Drift, length=0.5))
        el.append(env.new(f"d{block_id}.6",  xt.Drift, length=0.5))
        # Bend
        el.append(env.new(f"mb2.{block_id}.0", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb2.{block_id}.1", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb2.{block_id}.2", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb2.{block_id}.3", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb2.{block_id}.4", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb2.{block_id}.5", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        # Drift
        el.append(env.new(f"d{block_id}.7",  xt.Drift, length=0.5))
        el.append(env.new(f"d{block_id}.8",  xt.Drift, length=0.5))
        # QF
        el.append(env.new(f"mqf.{2*block_id}", xt.Quadrupole, length=0.3, k1=0.1))
        # Drift
        el.append(env.new(f"d{block_id}.9",   xt.Drift, length=0.5))
        el.append(env.new(f"d{block_id}.10",  xt.Drift, length=0.5))
        # Bend
        el.append(env.new(f"mb3.{block_id}.0", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb3.{block_id}.1", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb3.{block_id}.2", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb3.{block_id}.3", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb3.{block_id}.4", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb3.{block_id}.5", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        # Drift
        el.append(env.new(f"d{block_id}.11", xt.Drift, length=0.5))
        el.append(env.new(f"d{block_id}.12", xt.Drift, length=0.5))
        # QD
        el.append(env.new(f"mqd.{2*block_id}", xt.Quadrupole, length=0.3, k1=-0.7))
        # Drift
        el.append(env.new(f"d{block_id}.13", xt.Drift, length=0.5))
        el.append(env.new(f"d{block_id}.14", xt.Drift, length=0.5))
        # Bend
        el.append(env.new(f"mb4.{block_id}.0", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb4.{block_id}.1", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb4.{block_id}.2", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb4.{block_id}.3", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb4.{block_id}.4", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        el.append(env.new(f"mb4.{block_id}.5", xt.Bend, length=lbend, k0=k0_bend, h=h_bend))
        # Drift
        el.append(env.new(f"d{block_id}.15", xt.Drift, length=0.5))
        el.append(env.new(f"d{block_id}.16", xt.Drift, length=0.5))

        return el
    
    def add_bounding_apertures(name, idx, aperture, aper_dict, idx_aper):
        idx_aper.append(idx-1)
        aper_dict[name + '_aper_entry'] = aperture.copy()
        idx_aper.append(idx)
        aper_dict[name + '_aper_exit'] = aperture.copy()

    def insert_apertures(line, idx_aper, aper_dict):
        shift = 0
        for idx, (aper_name, aper) in zip(idx_aper, aper_dict.items()):
            adjusted_idx = idx + shift
            line.insert_element(at=adjusted_idx + 1, name=aper_name, element=aper)
            shift += 1

    # Create an environment
    env = xt.Environment()

    # Build ring
    n_blocks = int(round((2*np.pi) / (4*theta)))
    components = []
    for blk in range(1, n_blocks + 1):
        el = fodo_arc_block(blk)
        components += el

    line = env.new_line(components=components)
    line.particle_ref = xt.Particles(p0c=p0c, mass0=mass0, pdg_id=pdg_id)

    tab = line.get_table()

    # Install apertures
    aperture = xt.LimitRect(min_x=-aper_size, max_x=aper_size, min_y=-aper_size, max_y=aper_size)

    idx_aper = []
    aper_dict = {}
    needs_aperture = ['Bend', 'Quadrupole']
    for idx, (nn, ee) in enumerate(zip(tab.name, tab.element_type)):
        if ee in needs_aperture:
            add_bounding_apertures(nn, idx, aperture, aper_dict, idx_aper)

    insert_apertures(line, idx_aper, aper_dict)

    return line


def test_parameter_validation():
    line = build_toy_ring_with_apertures(aper_size=0.04)

    cases = [
        # (kwargs, expected_exc, regex)
        (dict(delta_negative_limit=0.0), ValueError, r"\bdelta_negative_limit must be < 0\b"),
        (dict(delta_positive_limit=0.0), ValueError, r"\bdelta_positive_limit must be > 0\b"),
        (dict(delta_step_size=0.0),      ValueError, r"\bdelta_step_size must be > 0\b"),
        (dict(s_start=1.0, s_end=1.0),   ValueError, r"\bs_start must be < s_end\b"),
        (dict(n_turns=0),                ValueError, r"\bn_turns must be > 0\b"),
        (dict(skip_elements=-1),         ValueError, r"\bskip_elements must be >= 0\b"),
        (dict(process_elements=0),       ValueError, r"\bprocess_elements must be > 0\b"),
    ]

    for kwargs, exc, pattern in cases:
        with pytest.raises(exc, match=pattern):
            line.momentum_aperture(
                nemitt_x=1e-6, nemitt_y=1e-6,
                **kwargs
            )


def test_forbid_resonance_crossing_not_implemented():
    line = build_toy_ring_with_apertures(aper_size=0.04)
    with pytest.raises(NotImplementedError, match=r"\bforbid_resonance_crossing is not implemented\b"):
        line.momentum_aperture(nemitt_x=1e-6, nemitt_y=1e-6, forbid_resonance_crossing=1)


def test_mutual_exclusivity_errors():
    line = build_toy_ring_with_apertures(aper_size=0.04)

    # x-plane conflict
    with pytest.raises(ValueError, match=r"Provide either x_offset or x_norm_offset"):
        line.momentum_aperture(
            nemitt_x=1e-6,
            nemitt_y=1e-6,
            x_offset=1e-3,
            x_norm_offset=1.0,
        )

    # y-plane conflict
    with pytest.raises(ValueError, match=r"Provide either y_offset or y_norm_offset"):
        line.momentum_aperture(
            nemitt_x=1e-6,
            nemitt_y=1e-6,
            y_offset=1e-3,
            y_norm_offset=1.0,
        )


def test_empty_selection_raises():
    line = build_toy_ring_with_apertures(aper_size=0.04)

    with pytest.raises(ValueError, match=r"No elements selected for momentum aperture computation."):
        line.momentum_aperture(
            nemitt_x=1e-6,
            nemitt_y=1e-6,
            include_name_pattern="does_not_exist_*",
            method="4d"
        )