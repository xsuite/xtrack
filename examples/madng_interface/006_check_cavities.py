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
madx.input('beam, particle=positron, gamma=20000;')
madx.use(sequence='myseq')
twmadx = xt.Table(madx.twiss(), _copy_cols=True)

# xsuite from madx source
line = xt.load(string=madx_src, format='madx').myseq
line.particle_ref = xt.Particles('positron', gamma0=20000)
twxs = line.twiss()

# twiss with MAD-NG from madx file
import pymadng as pg
mng = pg.MAD()
mng.send('''
    MADX:load('temp_seq.madx')
    MADX.myseq.beam = MAD.beam{particle='positron', gamma=20000};
    MAD.element.rfcavity.method = 2;
''')
twng_from_madx_df = mng.twiss(sequence='MADX.myseq')[0].to_df()
twng_from_madx = xt.Table({nn: twng_from_madx_df[nn].values for nn in ['name', 's', 'pt']}, _copy_cols=True)


import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1)
plt.plot(twmadx.s, twmadx.pt*1e3, '.-', label='madx')
plt.plot(twxs.s, twxs.ptau*1e3, 'x--', label='xsuite')
plt.plot(twng_from_madx.s, twng_from_madx.pt*1e3, 'o--', label='madng')

plt.legend()
plt.xlabel('s [m]')
plt.ylabel('pt [1e-3]')
plt.subplots_adjust(left=0.15)
plt.show()

