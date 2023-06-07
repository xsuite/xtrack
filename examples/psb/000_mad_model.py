from cpymad.madx import Madx

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
    call, file = 'psb.seq';
    call, file = 'psb_fb_lhc.str';
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
mad.input('''
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

# Handle bumpers (angle = 0)
seq = mad.sequence.psb1
seq.expanded_elements['bi1.bsw1l1.1'].angle = 1e-20
seq.expanded_elements['bi1.bsw1l1.2'].angle = 1e-20
seq.expanded_elements['bi1.bsw1l1.3'].angle = 1e-20
seq.expanded_elements['bi1.bsw1l1.4'].angle = 1e-20

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

k0BI1BSW1L11 := BSW_K0L/l_bsw1l1.1;
k0BI1BSW1L12 := -BSW_K0L/l_bsw1l1.2;
k0BI1BSW1L13 := -BSW_K0L/l_bsw1l1.3;
k0BI1BSW1L14 := BSW_K0L/l_bsw1l1.4;

bi1.bsw1l1.1, k2 := BSW_K2L/l_bsw1l1.1;
bi1.bsw1l1.2, k2 := -BSW_K2L/l_bsw1l1.2;
bi1.bsw1l1.3, k2 := -BSW_K2L/l_bsw1l1.3;
bi1.bsw1l1.4, k2 := BSW_K2L/l_bsw1l1.4;

''')


twmad = mad.twiss()


