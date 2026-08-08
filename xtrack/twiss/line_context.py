# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xobjects as xo

from .extra_markers import _build_auxiliary_tracker_with_extra_markers

import xtrack as xt  # To avoid circular imports


def _prepare_twiss_line_context(twiss_context, data):
    """Normalize range endpoints and enter temporary line/tracker state."""

    data = data.copy()
    line = data['line']

    if data['disable_apertures']:
        if not (line.tracker.track_flags.XS_FLAG_IGNORE_GLOBAL_APERTURE
                and line.tracker.track_flags.XS_FLAG_IGNORE_LOCAL_APERTURE):
            twiss_context.enter_context(xt.line._preserve_track_flags(line))
            line.tracker.track_flags.XS_FLAG_IGNORE_GLOBAL_APERTURE = True
            line.tracker.track_flags.XS_FLAG_IGNORE_LOCAL_APERTURE = True

    _resolve_open_twiss_range_defaults(data)
    _validate_multiturn_request(data)
    _resolve_twiss_range_endpoints(data)
    _set_twiss_periodic_mode(data)
    _enter_twiss_freeze_context(twiss_context, data)
    _prepare_4d_cavity_tracking(twiss_context, data)
    _prepare_twiss_at_s_markers(data)

    line = data['line']
    _prepare_radiation_tracking(twiss_context, data)

    if line.enable_time_dependent_vars:
        raise RuntimeError('Time dependent variables not supported in Twiss')

    return data


def _resolve_open_twiss_range_defaults(data):

    if ((data['init'] is not None
            or data['betx'] is not None
            or data['bety'] is not None)
            and data['start'] is None):
        data['start'] = xt.START
        data['end'] = data['end'] or xt.END


def _validate_multiturn_request(data):

    if data['num_turns'] == 1:
        return

    assert data['num_turns'] > 0
    assert data['start'] is None
    assert data['end'] is None
    assert data['init'] is None
    assert data['reverse'] is False


def _resolve_twiss_range_endpoints(data):

    data['start'] = _resolve_twiss_range_endpoint(
        line=data['line'], endpoint=data['start'], reverse=data['reverse'])
    data['end'] = _resolve_twiss_range_endpoint(
        line=data['line'], endpoint=data['end'], reverse=data['reverse'])


def _resolve_twiss_range_endpoint(line, endpoint, reverse):

    if endpoint is None:
        return None

    if isinstance(endpoint, xt.match._LOC):
        assert endpoint in [xt.START, xt.END]
        if reverse:
            endpoint = {xt.START: xt.END, xt.END: xt.START}[endpoint]
        endpoint = {
            xt.START: line._element_names_unique[0],
            xt.END: line._element_names_unique[-1],
        }[endpoint]

    assert isinstance(endpoint, str)  # index not supported anymore
    return endpoint


def _set_twiss_periodic_mode(data):

    init = data['init']
    if (init is not None and init not in ['periodic', 'periodic_symmetric']
            or data['betx'] is not None or data['bety'] is not None):
        data['periodic'] = False
        data['periodic_mode'] = None
        return

    data['periodic'] = True
    data['periodic_mode'] = init or 'periodic'
    for coordinate_name in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
        assert data[coordinate_name] is None, (
            f'``{coordinate_name}`` not supported for periodic twiss')


def _enter_twiss_freeze_context(twiss_context, data):

    line = data['line']
    if data['freeze_longitudinal']:
        twiss_context.enter_context(xt.freeze_longitudinal(line))
        data['freeze_longitudinal'] = False
    elif data['freeze_energy']:
        if not line._energy_is_frozen():
            twiss_context.enter_context(xt.line._preserve_config(line))
            line.freeze_energy(force=True)  # force is needed for collective lines
            data['freeze_energy'] = False


def _prepare_4d_cavity_tracking(twiss_context, data):

    line = data['line']
    if (data['method'] == '4d'
            and not line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK):
        twiss_context.enter_context(xt.line._preserve_track_flags(line))
        line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK = True


def _prepare_twiss_at_s_markers(data):

    if data['at_s'] is None:
        return

    if data['reverse']:
        raise NotImplementedError('``at_s`` not implemented for ``reverse``=True')
    if np.isscalar(data['at_s']):
        data['at_s'] = [data['at_s']]
    assert data['at_elements'] is None

    auxtracker, names_inserted_markers = (
        _build_auxiliary_tracker_with_extra_markers(
            tracker=data['line'].tracker,
            at_s=data['at_s'],
            marker_prefix='inserted_twiss_marker',
            algorithm='insert'))
    data['line'] = auxtracker.line
    data['at_elements'] = names_inserted_markers
    data['at_s'] = None
    data['strengths'] = True


def _prepare_radiation_tracking(twiss_context, data):

    line = data['line']
    radiation_method = data['radiation_method']

    if radiation_method is None and line._radiation_model is not None:
        if line._radiation_model in ('quantum', 'quantum-kick'):
            raise ValueError(
                'twiss cannot be called when the radiation model is stochastic')
        if data['method'] == '4d':
            raise RuntimeError(
                '4d twiss cannot be called when radiation is present')
        radiation_method = 'kick_as_co'
        data['radiation_method'] = radiation_method

    if radiation_method is not None and radiation_method != 'full':
        assert isinstance(line._context, xo.ContextCpu), (
            'Twiss with radiation computation is only supported on CPU')
        assert not line._context.openmp_enabled, (
            'Twiss with radiation computation is not supported with OpenMP '
            'parallelization')
        assert radiation_method in ['full', 'kick_as_co', 'scale_as_co']
        assert data['freeze_longitudinal'] is False

        if (radiation_method == 'kick_as_co'
                and not line.tracker.track_flags.XS_FLAG_SR_KICK_SAME_AS_FIRST):
            twiss_context.enter_context(xt.line._preserve_track_flags(line))
            line.tracker.track_flags.XS_FLAG_SR_KICK_SAME_AS_FIRST = True
        elif (radiation_method == 'scale_as_co'
                and (not hasattr(line.config, 'XTRACK_SYNRAD_SCALE_SAME_AS_FIRST')
                     or not line.config.XTRACK_SYNRAD_SCALE_SAME_AS_FIRST)):
            twiss_context.enter_context(xt.line._preserve_config(line))
            line.config.XTRACK_SYNRAD_SCALE_SAME_AS_FIRST = True

    if radiation_method == 'kick_as_co':
        assert line.tracker.track_flags.XS_FLAG_SR_KICK_SAME_AS_FIRST
