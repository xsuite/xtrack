import xtrack as xt

madx_src ='''
b1: sbend, l=1, angle=0.1;
q1: quadrupole, l=1, k1=0.01;
q2: quadrupole, l=1, k1=-0.01;
c1: rfcavity, l=0.1, lag=180/360, volt=4, freq=400;
c2: rfcavity, l=0.1, lag=-20/360, volt=3, freq=400;
c3: rfcavity, l=0.1, lag=(180-20)/360, volt=3, freq=400;
mm: marker;
myseq: sequence, l=10;
    b1, at=1;
    q1, at=2.5;
    c1, at=4;
    c2, at=5;
    c3, at=6;
    q2, at=7.5;
    mm, at=10;
endsequence;
'''

# write to file
with open('temp_seq.madx', 'w') as fid:
    fid.write(madx_src)

# twiss with madx directly
from cpymad.madx import Madx
madx = Madx()
madx.input(madx_src)
madx.input('beam, particle=positron, pc=10;')
madx.use(sequence='myseq')
twmadx = xt.Table(madx.twiss(), _copy_cols=True)

twmadx.cols['s pt']
# is
# Table: 16 rows, 3 cols
# name                      s            pt
# myseq$start:1             0  -2.73716e-13
# drift_0:0               0.5  -2.73716e-13
# b1:1                    1.5  -2.73716e-13
# drift_1:0                 2  -2.73716e-13
# q1:1                      3  -2.73716e-13
# drift_2:0              3.95  -2.73716e-13
# c1:1                   4.05  -2.73115e-13
# drift_3:0              4.95  -2.73115e-13
# c2:1                   5.05  -0.000102606 <-
# drift_4:0              5.95  -0.000102606 <-
# c3:1                   6.05  -2.73716e-13
# drift_5:0                 7  -2.73716e-13
# q2:1                      8  -2.73716e-13
# drift_6:0                10  -2.73716e-13
# mm:1                     10  -2.73716e-13

# twiss with MAD-NG from madx file
import pymadng as pg
mng = pg.MAD()
mng.send('''
    MADX:load('temp_seq.madx')
    MADX.myseq.beam = MAD.beam{particle='positron', pc=10};
    MADX.myseq:dumpseq()
''')
twng_from_madx_df = mng.twiss(sequence='MADX.myseq')[0].to_df()
twng_from_madx = xt.Table({nn: twng_from_madx_df[nn].values for nn in ['name', 's', 'pt']}, _copy_cols=True)

# 
madx2 = Madx()
madx2.input(line.to_madx_sequence(sequence_name='myseq'))
madx2.input('beam, particle=positron, pc=10;')
madx2.use(sequence='myseq')
twmadx2 = madx2.twiss()


env = xt.load(string=madx_src, format='madx')
line = env.myseq
line.particle_ref = xt.Particles('positron', p0c=10e9, q0=-1)
twxs = line.twiss()
mng_xs = line.to_madng()
mng_xs.seq.dumpseq()
tw = mng_xs.twiss(sequence=mng_xs.seq)[0].to_df()


