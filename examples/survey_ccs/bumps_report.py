"""Reader for the SU alignment (bump) request report.

The report is exported from the SU database with one row per *point*, not per
element: the name in the `Elément` column ends in `.E` (entrée, the element
start) or `.S` (sortie, the element end). Each of the four movable degrees of
freedom has a total and four additive contributions, `Beamb` (beam based),
`Cw`, `Mec` (mechanical) and `Opti` (optics), with a free-text comment each.

:func:`read_bumps_report` checks the file over and folds it into one row per
element, with the requested displacement of each end point in the element's
own RST frame. The RST component order matches ``xtrack._temp.survey_utils``,
whose functions call the two points start and end rather than entry and exit.
The report's positive radial deviation means motion along negative R
("Radial Position Deviation", section 5.4.2), so its radial values are negated
when converted to geometric RST displacements. S, T and roll are read as given.
"""

import pandas as pd

# Total of each degree of freedom, and the RST component it displaces.
TOTALS = {
    'r': 'Radial (m)',
    's': 'Longitudinal (m)',
    't': 'Vertical (m)',
}
ROLL = 'Roll (rad)'

# Every total is split over these four sources, which must add up to it. The
# roll totals are split over columns named after the tilt.
SOURCES = ['Beamb', 'Cw', 'Mec', 'Opti']
SOURCE_PREFIX = {
    'Radial (m)': 'Radial',
    'Longitudinal (m)': 'Longitudinal',
    'Vertical (m)': 'Vertical',
    'Roll (rad)': 'Tilt',
}

COLUMNS = ['r_entry', 's_entry', 't_entry',
           'r_exit', 's_exit', 't_exit',
           'roll', 'single_point']


def read_bumps_report(file_name):
    """Return one row per element, indexed by element name.

    The `*_entry` and `*_exit` columns hold the requested RST displacements of
    the two end points, in metres, and `roll` the requested roll about the
    chord, in radians. A blank cell in the report means no bump requested on
    that axis, and is read as zero.
    In particular, `r_entry` and `r_exit` are the negatives of the report's
    `Radial (m)` values, not the raw radial deviations.

    Some instruments have an entrance row only. To match GEODE, an unreported
    exit gets zero radial and vertical displacement. The different transverse
    displacements at the two ends then produce a crab: a centre socket moves
    by half the entrance transverse displacement. The longitudinal exit value
    is filled from the entrance as a placeholder; the rigid-body conversion
    determines the actual exit S from the chord length and transverse offsets.
    `single_point` flags the missing exit row, not the physical socket location
    (which the socket report specifies through its Origine and S columns).
    """
    raw = pd.read_csv(file_name)

    # The accented character in 'Elément' was lost in the csv export (it is a
    # literal U+FFFD replacement character in the file).
    raw = raw.rename(columns={'El�ment': 'Element'})

    _check_columns(raw)

    points = {}
    for _, row in raw.iterrows():
        name, point = row['Element'].rsplit('.', 1)
        if name != row['Nom Layout']:
            raise ValueError(
                f"{row['Element']}: name disagrees with Nom Layout "
                f"{row['Nom Layout']}")
        if point not in ('E', 'S'):
            raise ValueError(f"{row['Element']}: not an element end point")
        if (name, point) in points:
            raise ValueError(f"{row['Element']}: duplicate point")
        points[name, point] = row

    out = {}
    for name, point in points:
        if point != 'E':
            continue
        entry = points[name, 'E']
        exit_ = points.get((name, 'S'))

        # The roll is stored redundantly on both points of an element.
        roll = _value(entry, ROLL)
        if exit_ is not None and _value(exit_, ROLL) != roll:
            raise ValueError(f'{name}: roll differs between its two points')

        record = {'roll': roll, 'single_point': exit_ is None}
        for component, column in TOTALS.items():
            # Positive radial deviation is a displacement along negative R.
            sign = -1. if component == 'r' else 1.
            displacement = sign * _value(entry, column)
            record[f'{component}_entry'] = displacement
            if exit_ is None:
                record[f'{component}_exit'] = displacement if component == 's' else 0.
            else:
                record[f'{component}_exit'] = sign * _value(exit_, column)
        out[name] = record

    orphans = {name for name, point in points if point == 'S'} - set(out)
    if orphans:
        raise ValueError(f'exit point without an entrance: {sorted(orphans)}')

    return pd.DataFrame.from_dict(
        out, orient='index', columns=COLUMNS).rename_axis('name')


def _value(row, column):
    """Read one cell; a blank means no bump requested, i.e. zero."""
    value = row[column]
    return 0. if pd.isna(value) else float(value)


def _check_columns(raw):
    """Check each total is exactly the sum of its four source columns.

    This is what makes it safe to read only the totals and ignore the 32
    source and comment columns.
    """
    for column, prefix in SOURCE_PREFIX.items():
        sources = [f'{prefix} {ss}' for ss in SOURCES]
        missing = [cc for cc in [column] + sources if cc not in raw.columns]
        if missing:
            raise ValueError(f'missing columns: {missing}')
        residual = (raw[sources].fillna(0.).sum(axis=1)
                    - raw[column].fillna(0.)).abs()
        if residual.max() > 0:
            worst = raw['Element'][residual.idxmax()]
            raise ValueError(
                f'{column}: total is not the sum of {sources} '
                f'(worst {residual.max():.3e} at {worst})')
