import json
import numpy as np
from cpymad.madx import Madx

import xtrack as xt
import xdeps as xd
import xobjects as xo

import matplotlib.pyplot as plt

bumper_names = ['bi1.bsw1l1.1', 'bi1.bsw1l1.2', 'bi1.bsw1l1.3', 'bi1.bsw1l1.4']
thick_bumpers = {
'bi1.bsw1l1.1' : {'k0_name': 'k0BI1BSW1L11'},
'bi1.bsw1l1.2' : {'k0_name': 'k0BI1BSW1L12'},
'bi1.bsw1l1.3' : {'k0_name': 'k0BI1BSW1L13'},
'bi1.bsw1l1.4' : {'k0_name': 'k0BI1BSW1L14'},
}

mad = Madx()

# Load model
mad.input('''
call, file = '../../test_data/psb_chicane/psb.seq';
call, file = '../../test_data/psb_chicane/psb_fb_lhc.str';
beam, particle=PROTON, pc=0.5708301551893517;
use, sequence=psb1;
twiss;
''')

# Add shifts
mad.input('''
SELECT,FLAG=ERROR,CLEAR;
SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.1*;
EALIGN, DX=-0.0057;

SELECT,FLAG=ERROR,CLEAR;
SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.2*;
SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.3*;
SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.4*;
EALIGN, DX=-0.0442;
''')

# Match tunes
mad.input(f'''
QH = 4.4;
QV = 4.45;

MATCH, Sequence=psb1;
    VARY, NAME = kbrqf, STEP = 1e-3;
    VARY, NAME = kbrqd, STEP = 1e-3;
    GLOBAL, Q1 = QH;
    GLOBAL, Q2 = QV;
    JACOBIAN,CALLS=1000,TOLERANCE=1.0E-18,STRATEGY=3;
ENDMATCH;
''')

# # Handle bumpers (angle = 0)
# seq = mad.sequence.psb1
# seq.expanded_elements['bi1.bsw1l1.1'].angle = 1e-20
# seq.expanded_elements['bi1.bsw1l1.2'].angle = 1e-20
# seq.expanded_elements['bi1.bsw1l1.3'].angle = 1e-20
# seq.expanded_elements['bi1.bsw1l1.4'].angle = 1e-20

# Store bumpers length
for nn in bumper_names:
    thick_bumpers[nn]['length'] = mad.sequence.psb1.expanded_elements[nn].l

# Set K0 and K2 for thick bumpers
mad.input(f'''

l_bsw1l1.1 = {thick_bumpers['bi1.bsw1l1.1']['length']};
l_bsw1l1.2 = {thick_bumpers['bi1.bsw1l1.2']['length']};
l_bsw1l1.3 = {thick_bumpers['bi1.bsw1l1.3']['length']};
l_bsw1l1.4 = {thick_bumpers['bi1.bsw1l1.4']['length']};

BSW_K0L = 6.6E-2;
BSW_K2L = -9.7429e-02;
'''
)

mad.input('''
SELECT,FLAG=ERROR,CLEAR;
SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.1;
SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.4;
EFCOMP, DKN:={+BSW_K0L, 0, +BSW_K2L};
!EFCOMP, DKN:={0, 0, +BSW_K2L};

SELECT,FLAG=ERROR,CLEAR;
SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.2;
SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.3;
EFCOMP, DKN:={-BSW_K0L, 0, -BSW_K2L};

''')

# Twiss and ptc twiss with errors
twmad = mad.twiss()

beta0 = mad.sequence.psb1.beam.beta

twmad = twmad.dframe()

mad.input('''
ptc_create_universe;
ptc_create_layout, model=3, method=6, nst=5, exact=true;
ptc_setswitch, debuglevel=0, nocavity=false, fringe=true,
            exact_mis=true, time=true, totalpath=true;
select_ptc_normal, q1=0, q2=0;
select_ptc_normal, dq1=1, dq2=1;
select_ptc_normal, dq1=2, dq2=2;
PTC_ALIGN;
ptc_normal, closed_orbit, normal, icase=5, no=3;
ptc_twiss, closed_orbit, table=ptc_twiss, icase=4, no=3,
            summary_table=ptc_twiss_summary;
ptc_end;
''')
twptc = mad.table.ptc_twiss.dframe()
qx_ptc = mad.table.ptc_twiss.mu1[-1]
qy_ptc = mad.table.ptc_twiss.mu2[-1]
dqx_ptc = mad.table.normal_results.value[2] * beta0
dqy_ptc = mad.table.normal_results.value[3] * beta0
t_ptc = xd.Table(mad.table.ptc_twiss)

line_thick = xt.Line.from_json('psb_03_with_chicane_corrected.json')
line_thick.build_tracker()
line_thick.configure_bend_model(core='full', edge='full')
line_thick.vars['on_chicane_beta_corr'] = 0
line_thick.vars['on_chicane_tune_corr'] = 0

line_thin = xt.Line.from_json('psb_04_with_chicane_corrected_thin.json')
line_thin.build_tracker()
line_thin.vars['on_chicane_beta_corr'] = 0
line_thin.vars['on_chicane_tune_corr'] = 0

t_test = np.linspace(0, 6e-3, 100)

qx_thick = []
qy_thick = []
dqx_thick = []
dqy_thick = []
bety_at_scraper_thick = []
qx_thin = []
qy_thin = []
dqx_thin = []
dqy_thin = []
bety_at_scraper_thin = []

qx_ptc = []
qy_ptc = []
dqx_ptc = []
dqy_ptc = []
bety_at_scraper_ptc = []
for ii, tt in enumerate(t_test):
    print(f'Twiss at t = {tt*1e3:.2f} ms   ', end='\r', flush=True)
    line_thick.vars['t_turn_s'] = tt
    line_thin.vars['t_turn_s'] = tt

    tw_thick = line_thick.twiss()
    bety_at_scraper_thick.append(tw_thick['bety', 'br.stscrap22'])
    qx_thick.append(tw_thick.qx)
    qy_thick.append(tw_thick.qy)
    dqx_thick.append(tw_thick.dqx)
    dqy_thick.append(tw_thick.dqy)

    tw_thin = line_thin.twiss()
    bety_at_scraper_thin.append(tw_thin['bety', 'br.stscrap22'])
    qx_thin.append(tw_thin.qx)
    qy_thin.append(tw_thin.qy)
    dqx_thin.append(tw_thin.dqx)
    dqy_thin.append(tw_thin.dqy)

    mad.globals.bsw_k0l = line_thick.vars['bsw_k0l']._value
    mad.globals.bsw_k2l = line_thick.vars['bsw_k2l']._value

    mad.input('''
    SELECT,FLAG=ERROR,CLEAR;
    SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.1;
    SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.4;
    EFCOMP, DKN:={+BSW_K0L, 0, +BSW_K2L};
    !EFCOMP, DKN:={0, 0, +BSW_K2L};

    SELECT,FLAG=ERROR,CLEAR;
    SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.2;
    SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.3;
    EFCOMP, DKN:={-BSW_K0L, 0, -BSW_K2L};

    ''')

    mad.input('''
        ptc_create_universe;
        ptc_create_layout, model=3, method=6, nst=5, exact=true;
        ptc_setswitch, debuglevel=0, nocavity=false, fringe=true,
                    exact_mis=true, time=true, totalpath=true;
        select_ptc_normal, q1=0, q2=0;
        select_ptc_normal, dq1=1, dq2=1;
        select_ptc_normal, dq1=2, dq2=2;
        PTC_ALIGN;
        ptc_normal, closed_orbit, normal, icase=5, no=3;
        ptc_twiss, closed_orbit, table=ptc_twiss, icase=4, no=3,
                    summary_table=ptc_twiss_summary;
        ptc_end;
        ''')

    qx_ptc.append(mad.table.ptc_twiss.mu1[-1])
    qy_ptc.append(mad.table.ptc_twiss.mu2[-1])
    dqx_ptc.append(mad.table.normal_results.value[2] * beta0)
    dqy_ptc.append(mad.table.normal_results.value[3] * beta0)

    tw_ptc = xd.Table(mad.table.ptc_twiss)

    bety_at_scraper_ptc.append(tw_ptc['bety', 'br.stscrap22:1'])

qx_thick = np.array(qx_thick)
qy_thick = np.array(qy_thick)
dqx_thick = np.array(dqx_thick)
dqy_thick = np.array(dqy_thick)
bety_at_scraper_thick = np.array(bety_at_scraper_thick)
qx_thin = np.array(qx_thin)
qy_thin = np.array(qy_thin)
dqx_thin = np.array(dqx_thin)
dqy_thin = np.array(dqy_thin)
bety_at_scraper_thin = np.array(bety_at_scraper_thin)
qx_ptc = np.array(qx_ptc)
qy_ptc = np.array(qy_ptc)
dqx_ptc = np.array(dqx_ptc)
dqy_ptc = np.array(dqy_ptc)
bety_at_scraper_ptc = np.array(bety_at_scraper_ptc)

assert np.allclose(qx_thick, qx_ptc, atol=2e-4, rtol=0)
assert np.allclose(qy_thick, qy_ptc, atol=2e-4, rtol=0)
assert np.allclose(qx_thin, qx_ptc, atol=1e-3, rtol=0)
assert np.allclose(qy_thin, qy_ptc, atol=1e-3, rtol=0)
assert np.allclose(dqx_thick, dqx_ptc, atol=0.5, rtol=0)
assert np.allclose(dqy_thick, dqy_ptc, atol=0.5, rtol=0)
assert np.allclose(dqx_thin, dqx_ptc, atol=0.5, rtol=0)
assert np.allclose(dqy_thin, dqy_ptc, atol=0.5, rtol=0)
assert np.allclose(bety_at_scraper_thick, bety_at_scraper_ptc, atol=0, rtol=1e-2)
assert np.allclose(bety_at_scraper_thin, bety_at_scraper_ptc, atol=0, rtol=2e-2)

with open('ptc_ref.json', 'w') as fid:
    json.dump({
        't_test': t_test,
        'qx_ptc': qx_ptc,
        'qy_ptc': qy_ptc,
        'dqx_ptc': dqx_ptc,
        'dqy_ptc': dqy_ptc,
        'bety_at_scraper_ptc': bety_at_scraper_ptc,
    }, fid, cls=xo.JEncoder)




import matplotlib.pyplot as plt
plt.close('all')

fig1 = plt.figure(1)
sp1 = plt.subplot(2,1,1)

plt.plot(t_test*1e3, qy_thick, '-', color='C0', label='xsuite thick')
plt.plot(t_test*1e3, qy_thin, '-.', color='C1', label='xsuite thin')
plt.plot(t_test*1e3, qy_ptc, '--', color='C2', label='ptc')
plt.ylabel('tune')


plt.legend()

sp2 = plt.subplot(2,1,2, sharex=sp1)
plt.plot(t_test*1e3, bety_at_scraper_thick, '-', color='C0', label='xsuite thick')
plt.plot(t_test*1e3, bety_at_scraper_thin, '-.', label='xsuite thin')
plt.plot(t_test*1e3, bety_at_scraper_ptc, '--',label='ptc')
plt.ylim(bottom=0)

plt.legend()

plt.xlabel('time [ms]')
plt.ylabel(r'$\beta_y$ at scraper [m]')

plt.figure(2)
sp1 = plt.subplot(2,1,1, sharex=sp1)
plt.plot(t_test*1e3, dqx_thick, '-', color='C0', label='xsuite thick')
plt.plot(t_test*1e3, dqx_thin, '-.', color='C1', label='xsuite thin')
plt.plot(t_test*1e3, dqx_ptc, '--', color='C2', label='ptc')
plt.ylim(bottom=-9, top=0)
plt.ylabel(r"$Q'_x$")
plt.legend()

sp2 = plt.subplot(2,1,2, sharex=sp1)
plt.plot(t_test*1e3, dqy_thick, '-', color='C0', label='xsuite thick')
plt.plot(t_test*1e3, dqy_thin, '-.', color='C1', label='xsuite thin')
plt.plot(t_test*1e3, dqy_ptc, '--', color='C2', label='ptc')
plt.ylabel(r"$Q'_y$")
plt.ylim(bottom=-9,top=0)
plt.legend()

plt.xlabel('time [ms]')

plt.show()

