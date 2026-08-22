from cpymad.madx import Madx

madx_rbtrue = Madx()
madx_rbtrue.input('''
    b: rbend, l=1.0, angle=0.5;
    ss: sequence, l=5.0, refer=entry;
        b, at=2;
    endsequence;
    beam;
    use, sequence=ss;
    survey;
    twiss, betx=1, bety=1;
    ''')

tw_rb_true = madx_rbtrue.table.twiss.dframe()


madx_rbfalse = Madx()
madx_rbfalse.input('''
    option, -rbarc;
    b: rbend, l=1.0, angle=0.5;
    ss: sequence, l=5.0, refer=entry;
        b, at=2;
    endsequence;
    beam;
    use, sequence=ss;
    survey;
    twiss, betx=1, bety=1;
    ''')

tw_rb_false = madx_rbfalse.table.twiss.dframe()

madx_content = '''
    b: rbend, l=1.0, angle=0.5;
    ss: sequence, l=5.0, refer=entry;
        b, at=2;
    endsequence;
'''

with open('temp_rb.madx', 'w') as fid:
    fid.write(madx_content)

import pymadng as pg
mng = pg.MAD()
mng.send('''
    MADX:load('temp_rb.madx')
    MADX.ss.beam = MAD.beam{particle='proton', pc=26};
    MADX.ss:dumpseq()
''')
twng = mng.twiss(sequence='MADX.ss', betx=1, bety=1)[0].to_df()

print('madx rb true - default:')
print(tw_rb_true)
print('madx rb false:')
print(tw_rb_false)
print('madng:')
print(twng)