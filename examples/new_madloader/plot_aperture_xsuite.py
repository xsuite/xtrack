import matplotlib.pyplot as plt
import numpy as np

import xtrack as xt
from xtrack.mad_parser.loader import MadxLoader


# Load the LHC lattice with injection optics
# ==========================================

loader = MadxLoader(reverse_lines=['lhcb2'])
loader.load_file('EYETS 2023-2024.seq')
loader.load_file('INJECTION_EYETS 2023-2024.madx')
env = loader.env

particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)

lhcb1, lhcb2 = env.lines['lhcb1'], env.lines['lhcb2']
lhcb1.particle_ref = particle_ref
lhcb2.particle_ref = particle_ref

tw1 = lhcb1.twiss4d()
tw2 = lhcb2.twiss4d()

print('LHC thick:')
print(f'lhcb1: qx = {tw1.qx}, qy = {tw1.qy}')
print(f'lhcb2: qx = {tw2.qx}, qy = {tw2.qy}')


# Create a sliced lattice
# =======================

slicing_strategies = [
    xt.Strategy(None, name='.*'),
    xt.Strategy(xt.Teapot(4), name='m[bq].*'),
    xt.Strategy(xt.Teapot(16), name='mqx.*'),
    xt.Strategy(xt.Teapot(4), name='mb[xr].*'),
    xt.Strategy(xt.Teapot(4), name='mq[ym].*'),
    xt.Strategy(xt.Teapot(2), name='mqt.*'),
    xt.Strategy(None, element_type=xt.Solenoid),  # solenoid won't slice
]

lhcb1.replace_all_repeated_elements()
lhcb1.slice_thick_elements(slicing_strategies)

lhcb2.replace_all_repeated_elements()
lhcb2.slice_thick_elements(slicing_strategies)


# Build a lattice that contains apertures
# =======================================

# Will be needed later to generate a proper survey (we could also substitute
# apertures with markers at this point, it's not so important, but we'd have
# to change the script a little later on).

lhc_sequences = '''
    lhcb1: sequence;
    endsequence;
    lhcb2: sequence;
    endsequence;
'''

with open('APERTURE_EYETS 2023-2024.seq', 'r') as f:
    aperture_file = f.read()

input_string = lhc_sequences + aperture_file

aper_env = xt.Environment()
aper_loader = MadxLoader(env=aper_env, reverse_lines=['lhcb2'])
builders = aper_loader.load_string(input_string, build=False)
builder_ap1, builder_ap2 = builders

def aperture_line(line, builder):
    # The aperture file expects there to already be some markers in the line,
    # so we add them here.
    print(f'Building aperture-only version of {line.name}')

    relational_markers = set(p.from_ for p in builder.components)

    for ip in relational_markers:
        builder.new(ip, 'marker', at=line.get_table().rows[ip].s[0])

    return builder.build()

aper1 = aperture_line(lhcb1, builder_ap1)
aper2 = aperture_line(lhcb2, builder_ap2)

# Insert apertures into lattice

def insert_apertures(line, apertures):
    print(f'Inserting apertures into {line.name}')
    tt = apertures.get_table()
    apertures = tt.rows[tt.element_type != 'Marker']
    apertures = apertures.rows[apertures.element_type != 'Drift']

    line._insert_thin_elements_at_s(elements_to_insert=[
        (row.s, [(row.name, aper1[row.name])]) for row in apertures.cols['name', 's'].rows
        if not row.name == '_end_point'
    ])

insert_apertures(lhcb1, aper1)
insert_apertures(lhcb2, aper2)

# Twiss thin to verify
# ====================

tw1_thin = lhcb1.twiss4d()
tw2_thin = lhcb2.twiss4d()

print('After slicing:')
print(f'lhcb1: qx = {tw1_thin.qx}, qy = {tw1_thin.qy}')
print(f'lhcb2: qx = {tw2_thin.qx}, qy = {tw2_thin.qy}')

# Straight surveys
# ================

arc_vars = [f'ab.a{ab}' for ab in ['12', '23', '34', '45', '56', '67', '78', '81']]
old_vars = {}
for var in arc_vars:
    old_vars[var] = env.vars[var]
    env.vars[var] = 0

sv1 = lhcb1.survey()
sv2 = lhcb2.survey()


# Compute offsets
# ===============

def offset_elements(line, survey, dir=1):
    tt = line.get_table()
    apertypes = ['LimitEllipse', 'LimitRect', 'LimitRectEllipse', 'LimitRacetrack']
    aper_idx = np.isin(tt.element_type, apertypes)
    for row in tt.rows[aper_idx].rows:
        el = line[row.name]
        mech_sep = el.extra['mech_sep'] * dir
        x = survey.rows[row.name].X
        el.shift_x = mech_sep / 2 - x

    return aper_idx

# Convenience function to compute aperture size and beam sizes
# ============================================================

def get_aperture_size(el):
    if hasattr(el, 'min_x'):
        return el.min_x, el.max_x
    if hasattr(el, 'max_x'):
        return -el.max_x, el.max_x
    return -el.a, el.a


def compute_beam_size(survey, twiss):
    sx = survey.X
    s = twiss.s
    x = twiss.x
    bx = twiss.betx
    dx = twiss.dx
    sigx = 13 * np.sqrt(2.5e-6 / 450 * 0.938 * bx) + abs(dx) * 8e-4

    return s, sx, x, sigx


# Make plots
# ==========

def plot_apertures(line, twiss, survey):
    aper_idx = offset_elements(line, survey, dir=1 if line.name == 'lhcb1' else -1)

    tw_ap = twiss.rows[aper_idx]
    sv_ap = survey.rows[aper_idx]
    ap_extent = np.array([get_aperture_size(line[nn]) for nn in tw_ap.name])
    ap_offset = np.array([line[nn].shift_x for nn in tw_ap.name])

    upper = ap_offset + ap_extent[:, 0] + sv_ap.X
    lower = ap_offset + ap_extent[:, 1] + sv_ap.X

    plt.fill_between(tw_ap.s, upper, lower, alpha=0.5, color="k")
    plt.plot(tw_ap.s, upper, color="k")
    plt.plot(tw_ap.s, lower, color="k")


def plot_beam_size(twiss, survey, color):
    s, sx, x, sigx = compute_beam_size(survey, twiss)
    plt.fill_between(s, x - sigx + sx, x + sigx + sx, alpha=0.5, color=color)


plot_apertures(lhcb1, tw1_thin, sv1)
plot_apertures(lhcb2, tw2_thin, sv2)

plot_beam_size(tw1_thin, sv1, color='b')
plot_beam_size(tw2_thin, sv2, color='r')
