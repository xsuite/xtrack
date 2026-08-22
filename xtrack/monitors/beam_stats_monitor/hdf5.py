import numpy as np


def save_to_file(monitor, output_file=None):
    if output_file is not None:
        monitor._output_file = output_file

    if monitor._output_file is None:
        return

    h5py = _import_h5py()
    with h5py.File(monitor._output_file, 'a') as h5file:
        _initialize_or_validate_hdf5_file(monitor, h5file)

        local_start = _get_local_start_index_from_hdf5(monitor, h5file)
        local_stop = monitor._num_touched_records()
        if local_stop <= local_start:
            return

        record_slice = slice(local_start, local_stop)
        _append_hdf5_dataset(h5file, 'turns', monitor.turns[record_slice])

        stats_group = h5file.require_group('stats')
        for level in monitor.available_levels:
            level_group = stats_group.require_group(level)
            for stat in monitor.stats:
                _append_hdf5_dataset(
                    level_group, stat,
                    monitor.get(stat, level=level)[record_slice])

        if len(monitor.profile_coordinates) > 0:
            profiles_group = h5file.require_group('profiles')
            for coord in monitor.profile_coordinates:
                profile_group = profiles_group.require_group(coord)
                _append_hdf5_dataset(
                    profile_group, 'counts',
                    monitor.profiles[coord][record_slice])

        h5file.flush()


def initialize_output_file(monitor):
    h5py = _import_h5py()
    with h5py.File(monitor._output_file, 'w') as h5file:
        _initialize_hdf5_file(monitor, h5file)
        h5file.flush()


def _import_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover
        msg = 'h5py is required for BeamStatsMonitor HDF5 output'
        raise ModuleNotFoundError(msg) from exc
    return h5py


def _get_local_start_index_from_hdf5(monitor, h5file):
    if 'turns' not in h5file or len(h5file['turns']) == 0:
        return 0

    last_turn = int(h5file['turns'][-1])
    turns = monitor.turns
    if len(turns) == 0:
        return 0

    matches = np.nonzero(turns == last_turn)[0]
    if len(matches) > 0:
        return int(matches[0]) + 1

    if last_turn < turns[0]:
        return 0

    raise RuntimeError(
        'Cannot append BeamStatsMonitor frame because the output file '
        'already contains turns beyond the current frame')


def _initialize_or_validate_hdf5_file(monitor, h5file):
    if 'schema_version' not in h5file.attrs:
        if len(h5file.keys()) != 0:
            raise ValueError(
                'Output HDF5 file is not empty and does not contain '
                'BeamStatsMonitor metadata')
        _initialize_hdf5_file(monitor, h5file)
    else:
        _validate_hdf5_file(monitor, h5file)


def _initialize_hdf5_file(monitor, h5file):
    h5file.attrs['schema_version'] = 1
    h5file.attrs['class'] = 'BeamStatsMonitor'
    h5file.attrs['stats'] = np.array(monitor.stats, dtype='S')
    h5file.attrs['available_levels'] = np.array(
        monitor.available_levels, dtype='S')
    h5file.attrs['default_level'] = monitor.default_level
    h5file.attrs['every_n_turns'] = int(monitor.every_n_turns)
    h5file.attrs['n_records_per_frame'] = int(monitor._num_records)
    h5file.attrs['coasting'] = monitor.coasting
    h5file.attrs['particle_id_range'] = np.array(
        [-1, -1] if monitor.particle_id_range is None
        else monitor.particle_id_range,
        dtype=np.int64)
    h5file.attrs['profile_coordinates'] = np.array(
        monitor.profile_coordinates, dtype='S')

    if not monitor.coasting:
        h5file.create_dataset(
            'filled_slots', data=monitor.filled_slots.astype(np.int64))
        h5file.create_dataset(
            'selected_slots', data=monitor.selected_slots.astype(np.int64))
    if monitor.zeta_centers is not None:
        h5file.create_dataset('zeta_centers', data=monitor.zeta_centers)

    if len(monitor.profile_coordinates) > 0:
        profiles_group = h5file.create_group('profiles')
        profile_edges = monitor.profile_bin_edges
        profile_centers = monitor.profile_bin_centers
        for coord in monitor.profile_coordinates:
            profile_group = profiles_group.create_group(coord)
            profile_group.create_dataset(
                'bin_edges', data=profile_edges[coord])
            profile_group.create_dataset(
                'bin_centers', data=profile_centers[coord])


def _validate_hdf5_file(monitor, h5file):
    expected_attrs = {
        'schema_version': 1,
        'class': 'BeamStatsMonitor',
        'stats': np.array(monitor.stats, dtype='S'),
        'available_levels': np.array(monitor.available_levels, dtype='S'),
        'default_level': monitor.default_level,
        'every_n_turns': int(monitor.every_n_turns),
        'n_records_per_frame': int(monitor._num_records),
        'profile_coordinates': np.array(
            monitor.profile_coordinates, dtype='S'),
    }
    for name, expected in expected_attrs.items():
        if name not in h5file.attrs:
            raise ValueError(
                f'Output HDF5 file is missing metadata `{name}`')
        actual = h5file.attrs[name]
        if isinstance(expected, np.ndarray):
            if not np.array_equal(np.asarray(actual), expected):
                raise ValueError(
                    f'Output HDF5 metadata `{name}` does not match '
                    'this monitor')
        elif actual != expected:
            raise ValueError(
                f'Output HDF5 metadata `{name}` does not match this '
                'monitor')

    actual_coasting = bool(h5file.attrs.get('coasting', False))
    if actual_coasting != monitor.coasting:
        raise ValueError(
            'Output HDF5 metadata `coasting` does not match this monitor')

    expected_particle_id_range = np.array(
        [-1, -1] if monitor.particle_id_range is None
        else monitor.particle_id_range,
        dtype=np.int64)
    if 'particle_id_range' in h5file.attrs:
        if not np.array_equal(
                np.asarray(h5file.attrs['particle_id_range']),
                expected_particle_id_range):
            raise ValueError(
                'Output HDF5 metadata `particle_id_range` does not match '
                'this monitor')
    elif monitor.particle_id_range is not None:
        raise ValueError(
            'Output HDF5 file is missing metadata `particle_id_range`')

    if not monitor.coasting:
        _check_hdf5_dataset_equal(
            h5file, 'filled_slots', monitor.filled_slots.astype(np.int64))
        _check_hdf5_dataset_equal(
            h5file, 'selected_slots', monitor.selected_slots.astype(np.int64))

    if len(monitor.profile_coordinates) == 0:
        if 'profiles' in h5file:
            raise ValueError(
                'Output HDF5 file contains profiles but this monitor does not')
    else:
        if 'profiles' not in h5file:
            raise ValueError('Output HDF5 file is missing group `profiles`')
        for coord in monitor.profile_coordinates:
            profile_path = f'profiles/{coord}'
            if profile_path not in h5file:
                raise ValueError(
                    f'Output HDF5 file is missing group `{profile_path}`')
            _check_hdf5_dataset_equal(
                h5file, f'{profile_path}/bin_edges',
                monitor.profile_bin_edges[coord])
            _check_hdf5_dataset_equal(
                h5file, f'{profile_path}/bin_centers',
                monitor.profile_bin_centers[coord])


def _check_hdf5_dataset_equal(group, name, expected):
    if name not in group:
        raise ValueError(f'Output HDF5 file is missing dataset `{name}`')
    if not np.array_equal(group[name][...], expected):
        raise ValueError(
            f'Output HDF5 dataset `{name}` does not match this monitor')


def _append_hdf5_dataset(group, name, data):
    data = np.asarray(data)
    tail_shape = data.shape[1:]
    if name in group:
        dataset = group[name]
        if dataset.shape[1:] != tail_shape:
            raise ValueError(
                f'Output HDF5 dataset `{dataset.name}` has shape '
                f'{dataset.shape}, expected tail shape {tail_shape}')
    else:
        dataset = group.create_dataset(
            name,
            shape=(0, *tail_shape),
            maxshape=(None, *tail_shape),
            chunks=(1, *tail_shape),
            dtype=data.dtype)
    old_size = dataset.shape[0]
    new_size = old_size + data.shape[0]
    dataset.resize((new_size, *dataset.shape[1:]))
    dataset[old_size:new_size] = data
