import xtrack as xt

madx_src ='''

kf = 0.01;
kd = -0.01;

b1: sbend, l=1, angle=0.1;
q1: quadrupole, l=1, k1:=kf;
q2: quadrupole, l=1, k1:=kd;
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

# twiss with MAD-NG from madx file
import pymadng as pg
mng = pg.MAD()
mng.send(r'''
    MADX:load('temp_seq.madx')
    MADX.myseq.beam = MAD.beam{particle='positron', gamma=20000};
    MAD.element.rfcavity.method = 2;

    local tw = MAD.twiss{sequence=MADX.myseq}
    tw:print({'name', 's', 'beta11'})

    local obs_flag = MAD.element.flags.observed

    MADX.myseq:select(obs_flag, {list={'C1', 'MM'}})

    local X0 = MAD.damap{nv=6, m0=3}
    local trk, mflw = MAD.track{sequence=MADX.myseq, X0=X0, savemap=true}
    trk:print({'name', 's', 'x', 'px', 'y', 'py', 't', 'pt'})

    -- mflw[1]:print() -- map at the end

    trk['MM'].__map:print() -- map at the end
''')

# twng_from_madx_df = mng.twiss(sequence='MADX.myseq')[0].to_df()
# twng_from_madx = xt.Table({nn: twng_from_madx_df[nn].values for nn in ['name', 's', 'pt']}, _copy_cols=True)

