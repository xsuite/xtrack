import numpy as np
from madpoint import MadPoint

_sigma_names = [11, 12, 13, 14, 22, 23, 24, 33, 34, 44]
_beta_names = ["betx", "bety"]


def norm(v):
    return np.sqrt(np.sum(v ** 2))


def get_points_twissdata_for_elements(
    ele_names, mad, seq_name, use_survey=True, use_twiss=True
):

    mad.use(sequence=seq_name)
    mad.twiss()

    if use_survey:
        mad.survey()

    bb_xyz_points = []
    bb_twissdata = {
        kk: []
        for kk in _sigma_names
        + _beta_names
        + "dispersion_x dispersion_y x y".split()
    }
    for eename in ele_names:
        bb_xyz_points.append(
            MadPoint(
                eename + ":1", mad, use_twiss=use_twiss, use_survey=use_survey
            )
        )

        i_twiss = np.where(mad.table.twiss.name == (eename + ":1"))[0][0]

        for sn in _sigma_names:
            bb_twissdata[sn].append(
                getattr(mad.table.twiss, "sig%d" % sn)[i_twiss]
            )

        for kk in ["betx", "bety"]:
            bb_twissdata[kk].append(mad.table.twiss[kk][i_twiss])
        gamma = mad.table.twiss.summary.gamma
        beta = np.sqrt(1.0 - 1.0 / (gamma * gamma))
        for pp in ["x", "y"]:
            bb_twissdata["dispersion_" + pp].append(
                mad.table.twiss["d" + pp][i_twiss] * beta
            )
            bb_twissdata[pp].append(mad.table.twiss[pp][i_twiss])
        # , 'dx', 'dy']:

    return bb_xyz_points, bb_twissdata


def get_elements(seq, ele_type=None, slot_id=None):

    elements = []
    element_names = []
    for ee in seq.elements:

        if ele_type is not None:
            if ee.base_type.name != ele_type:
                continue

        if slot_id is not None:
            if ee.slot_id != slot_id:
                continue

        elements.append(ee)
        element_names.append(ee.name)

    return elements, element_names


def get_points_twissdata_for_element_type(
    mad, seq_name, ele_type=None, slot_id=None, use_survey=True, use_twiss=True
):

    elements, element_names = get_elements(
        seq=mad.sequence[seq_name], ele_type=ele_type, slot_id=slot_id
    )

    points, twissdata = get_points_twissdata_for_elements(
        element_names,
        mad,
        seq_name,
        use_survey=use_survey,
        use_twiss=use_twiss,
    )

    return elements, element_names, points, twissdata



##################################
# space charge related functions #
##################################
sc_mode_to_slotid = {"Coasting": "1", "Bunched": "2", "Interpolated": "3"}


def determine_sc_locations(line, n_SCkicks, length_fuzzy):
    s_elements = np.array(line.get_s_elements())
    length_target = s_elements[-1] / float(n_SCkicks)
    s_targets = np.arange(0, s_elements[-1], length_target)
    sc_locations = []
    for s in s_targets:
        idx_closest = (np.abs(s_elements - s)).argmin()
        s_closest = s_elements[idx_closest]
        if abs(s - s_closest) < length_fuzzy / 2.0:
            sc_locations.append(s_closest)
        else:
            sc_locations.append(s)
    sc_lengths = np.diff(sc_locations).tolist() + [
        s_elements[-1] - sc_locations[-1]
    ]
    return sc_locations, sc_lengths


def install_sc_placeholders(mad, seq_name, name, s, mode="Bunched"):
    sid = sc_mode_to_slotid[mode]
    mad.input(
        f"""
            seqedit, sequence={seq_name};"""
    )
    for name_, s_ in zip(np.atleast_1d(name), np.atleast_1d(s)):
        mad.input(
            f"""
            {name_} : placeholder, l=0., slot_id={sid};
            install, element={name_}, at={s_:.10e};"""
        )
    mad.input(
        f"""
            flatten;
            endedit;
            use, sequence={seq_name};"""
    )


def get_spacecharge_names_twdata(mad, seq_name, mode):
    _, mad_sc_names, _, twdata = get_points_twissdata_for_element_type(
        mad,
        seq_name,
        ele_type="placeholder",
        slot_id=int(sc_mode_to_slotid[mode]),
        use_survey=False,
        use_twiss=True,
    )
    return mad_sc_names, twdata


def _setup_spacecharge_in_line(
    sc_elements,
    sc_lengths,
    sc_twdata,
    betagamma,
    number_of_particles,
    delta_rms,
    neps_x,
    neps_y,
):

    for ii, ss in enumerate(sc_elements):
        ss.longitudinal_profile.number_of_particles = number_of_particles
        ss.sigma_x = np.sqrt(
            sc_twdata["betx"][ii] * neps_x / betagamma
            + (sc_twdata["dispersion_x"][ii] * delta_rms) ** 2
        )
        ss.sigma_y = np.sqrt(
            sc_twdata["bety"][ii] * neps_y / betagamma
            + (sc_twdata["dispersion_y"][ii] * delta_rms) ** 2
        )
        ss.length = sc_lengths[ii]
        ss.mean_x = sc_twdata["x"][ii]
        ss.mean_y = sc_twdata["y"][ii]


def setup_spacecharge_bunched_in_line(
    sc_elements,
    sc_lengths,
    sc_twdata,
    betagamma,
    number_of_particles,
    delta_rms,
    neps_x,
    neps_y,
    bunchlength_rms,
):

    for ii, ss in enumerate(sc_elements):
        ss.longitudinal_profile.sigma_z = bunchlength_rms
    _setup_spacecharge_in_line(
        sc_elements,
        sc_lengths,
        sc_twdata,
        betagamma,
        number_of_particles,
        delta_rms,
        neps_x,
        neps_y,
    )


def setup_spacecharge_coasting_in_line(
    sc_elements,
    sc_lengths,
    sc_twdata,
    betagamma,
    number_of_particles,
    delta_rms,
    neps_x,
    neps_y,
    circumference,
):

    for ii, ss in enumerate(sc_elements):
        ss.circumference = circumference
    _setup_spacecharge_in_line(
        sc_elements,
        sc_lengths,
        sc_twdata,
        betagamma,
        number_of_particles,
        delta_rms,
        neps_x,
        neps_y,
    )


def setup_spacecharge_interpolated_in_line(
    sc_elements,
    sc_lengths,
    sc_twdata,
    betagamma,
    number_of_particles,
    delta_rms,
    neps_x,
    neps_y,
    line_density_profile,
    dz,
    z0,
    method=0,
):
    assert method == 0 or method == 1
    for ii, ss in enumerate(sc_elements):
        ss.line_density_profile = line_density_profile
        ss.dz = dz
        ss.z0 = z0
        ss.method = method
    _setup_spacecharge_in_line(
        sc_elements,
        sc_lengths,
        sc_twdata,
        betagamma,
        number_of_particles,
        delta_rms,
        neps_x,
        neps_y,
    )


def check_spacecharge_consistency(
    sc_elements, sc_names, sc_lengths, mad_sc_names
):
    assert len(sc_elements) == len(mad_sc_names)
    assert len(sc_lengths) == len(mad_sc_names)
    for ii, (ss, nn) in enumerate(zip(sc_elements, sc_names)):
        assert nn == mad_sc_names[ii]
