import numpy as np
import pandas as pd

# Alignment (bump) request report exported from the SU database. Each row is a
# *point*, not an element: the element name carries a suffix `.E` (entrée, i.e.
# element start) or `.S` (sortie, i.e. element end), matching the RST start/end
# offsets used in `xtrack._temp.survey_utils`.
df = pd.read_csv('Bumps_sp_report.csv')

# The accented character in 'Elément' was lost in the csv export (it is a
# literal U+FFFD replacement character in the file)
df.rename(columns={'El�ment': 'Element'}, inplace=True)

df['layout'] = df['Element'].str.rsplit('.', n=1).str[0]
df['point'] = df['Element'].str.rsplit('.', n=1).str[1]
assert (df['layout'] == df['Nom Layout']).all()

# The four movable degrees of freedom. Each has a total, a date, and four
# additive contributions (Beamb = beam-based, Cw = cold/warn?, Mec = mechanical,
# Opti = optics) with an associated free-text comment column.
DOFS = {
    'Radial (m)': 'Radial',        # R: radial offset
    'Longitudinal (m)': 'Longitudinal',  # S: longitudinal offset (along chord)
    'Vertical (m)': 'Vertical',    # T: transverse (vertical) offset
    'Roll (rad)': 'Tilt',          # roll about the chord
}
SOURCES = ['Beamb', 'Cw', 'Mec', 'Opti']

print(f'{len(df)} rows ({df["layout"].nunique()} distinct elements), '
      f'{len(df.columns) - 2} columns\n')

print('=== column blocks ===')
for ii, name in enumerate(df.columns.drop(['layout', 'point'])):
    print(f'{ii:3d}  {name}')

print('\n=== the total of each dof is the sum of its four sources ===')
for total, base in DOFS.items():
    parts = [f'{base} {ss}' for ss in SOURCES]
    residual = (df[parts].fillna(0).sum(axis=1) - df[total].fillna(0)).abs()
    print(f'{total:20s} max|sum(sources) - total| = {residual.max():.1e}')

print('\n=== how many points request each dof ===')
for total, base in DOFS.items():
    per_source = ', '.join(
        f'{ss}={df[f"{base} {ss}"].notna().sum()}' for ss in SOURCES)
    print(f'{total:20s} {df[total].notna().sum():3d} points  ({per_source})')

# Pivot the .E/.S point rows into one row per element, which is the form
# consumed by su.misalignment_from_rst_displacements: the two end-point
# displacements plus the roll about the chord (which cannot be inferred from
# the end points).
piv = {}
for total in DOFS:
    piv[total] = df.pivot_table(index='layout', columns='point', values=total,
                                dropna=False).reindex(columns=['E', 'S'])

elements = piv['Radial (m)'].index

# RST component order follows survey_utils: E_rst = column_stack((er, es, et)),
# i.e. (radial, longitudinal, transverse). These are displacements from the
# nominal positions, not absolute offsets.
displ_start_rst = np.column_stack([piv[kk]['E'].values for kk in
                                    ('Radial (m)', 'Longitudinal (m)',
                                     'Vertical (m)')])
displ_end_rst = np.column_stack([piv[kk]['S'].values for kk in
                                  ('Radial (m)', 'Longitudinal (m)',
                                   'Vertical (m)')])
roll = piv['Roll (rad)']['E'].values

# Roll is stored redundantly on both points; check it is consistent before
# using the entrance value as the single per-element `bgamma`.
roll_es = piv['Roll (rad)']
both = roll_es.notna().all(axis=1)
assert np.allclose(roll_es['E'][both], roll_es['S'][both], atol=0, rtol=0)

# Elements with no .S *row* are the zero-length instruments (scintillators,
# wire chambers): a single point fully defines their position. Note that a NaN
# in a pivoted dof only means that dof was not requested at that point, so the
# missing points must be found from the rows themselves.
points = df.groupby('layout')['point'].apply(lambda ss: ''.join(sorted(ss)))
only_start = points.reindex(elements) == 'E'
print(f'\n=== {only_start.sum()} elements have only an .E point '
      f'(zero-length instruments) ===')
print('  ' + ', '.join(elements[only_start.values]))

print('\n=== E/S pattern per dof (over elements requesting it) ===')
for total in DOFS:
    pp = piv[total].dropna(how='all')
    ee, ss = pp['E'], pp['S']
    sym = ((ee - ss).abs() < 1e-15).sum()          # translation, stays parallel
    crab = (((ee + ss).abs() < 1e-15) & (ee.abs() > 0)).sum()  # pure rotation
    print(f'{total:20s} n={len(pp):3d}  symmetric(E=S)={sym:3d}  '
          f'crab(E=-S)={crab:3d}  S-blank={ss.isna().sum():3d}  '
          f'|max|={np.nanmax(np.abs(pp.values)):.4f}')

# The Mec comments label each request with the test case it belongs to.
comments = pd.concat([df[f'Comm {tag} Mec'] for tag in 'RVTL']).dropna()
kinds = comments.str.extract(r'^(Test [^:]+):')[0].value_counts()
print('\n=== test cases declared in the comments ===')
print(kinds.to_string())

print('\n=== per-element RST displacements (first 12) ===')
out = pd.DataFrame({
    'R_E': displ_start_rst[:, 0], 'S_E': displ_start_rst[:, 1],
    'T_E': displ_start_rst[:, 2],
    'R_S': displ_end_rst[:, 0], 'S_S': displ_end_rst[:, 1],
    'T_S': displ_end_rst[:, 2],
    'roll': roll,
}, index=elements)
print(out.head(12).to_string())
