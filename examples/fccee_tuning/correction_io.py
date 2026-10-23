"""Save the FCC-ee correction result and metadata."""


def save_result(line, path, *, tw_ref, progress, completed, metrics, settings):
    """Save an independently reloadable line with correction provenance."""
    circuits = line.metadata['correction_knobs']
    metadata = {
        'method': '4d',
        'delta0': 0.0,
        'ramp_schedule': 'equal_fraction',
        'nonlinear_ramp_knob': 'on_nonlinear',
        'nonlinear_ramp_min_order': 2,
        'nominal_reference': 'input model with on_errors=0',
        'orbit_response': 'aligned 4D optics at delta=0',
        'optics_response': 'aligned thick-quadrupole beta/alpha/Dx/tune response',
        'tune_response': 'fixed aligned 2x2 response',
        'orbit_trials': 'only_orbit for intermediate steering; ordinary full Twiss before optics/coupling checks',
        'fitted_optics': ['betx', 'bety', 'alfx', 'alfy', 'dx', 'qx', 'qy', 'f1001', 'f1010', 'dy'],
        'skew_trim_unit': 'm^-1',
        'quadrupole_trim_prefix': 'dk1_optics_',
        'quadrupole_trim_unit': 'm^-2',
    }
    metadata.update(settings)
    metadata.update(
        error_fraction=float(line['on_errors']),
        nonlinear_fraction=float(line['on_nonlinear']),
        progress=progress,
        completed=bool(completed),
        target_tunes=[float(tw_ref.qx), float(tw_ref.qy)],
        nonlinear_ramp_magnets=circuits['nonlinear_magnets'],
        nominal_tune_knobs=circuits['tunes']['nominal_values'],
        added_correctors=circuits['added_correctors'],
        horizontal_dispersion_rms_m=metrics['dx_rms'],
        horizontal_dispersion_ip_max_m=metrics['dx_ip'],
        dispersion_rms_m=metrics['dy_rms'],
        dispersion_ip_max_m=metrics['dy_ip'],
        coupling_rms=metrics['coupling_rms'].tolist(),
        coupling_ip_max=float(metrics['coupling_ip']),
        bpm_orbit_rms_m=metrics['orbit_rms'].tolist(),
        c_minus=metrics['c_minus'],
    )
    line.metadata['orbit_optics_correction'] = metadata
    line.to_json(path, indent=None)
