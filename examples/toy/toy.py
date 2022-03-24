import numpy as np
from matplotlib import pyplot as plt
from cpymad.madx import Madx

import xtrack as xt
import xpart as xp

import sys
sys.path.append('/afs/cern.ch/user/x/xbuffat/harpy')
from harmonic_analysis import HarmonicAnalysis

mad = Madx()
mad.option(echo=True, debug=False, info=False, warn=False)

mad.input('call file="toy.seq";')
mad.input('Beam, particle = positron, sequence=toy, energy = 20.0;')
mad.input('use, sequence=toy;')
mad.input('makethin;')
mad.input('select,flag=twiss,column=name,s,x,y,mux,betx,muy,bety,dx,dy;')

twiss0 = mad.twiss().copy()
summ0  = mad.table.summ.copy()

use_mat = True
if use_mat:
    dqx_mat = 0.01
    dqy_mat = 0.01

    mad.input(
f"""
dqx_mat = {dqx_mat};
dqy_mat = {dqy_mat};
twopidphix := {2.0*np.pi}*dqx_mat;
twopidphiy := {2.0*np.pi}*dqy_mat;
""")

    mad.input(
"""
PHASESHIFT(seqname, LOC, FROMRK, twopidphix, twopidphiy) : macro = {

markFROMRKLOC: marker;
seqedit,sequence=seqname;
remove, element=MATFROMRKLOC;
remove, element=markFROMRKLOC;
endedit;

seqedit,sequence=seqname;
install,element=markFROMRKLOC,at=LOC,from=FROMRK;
endedit;

MATFROMRKLOC : MATRIX, RM11:=R11FROMRKLOC,RM12:=R12FROMRKLOC,RM21:=R21FROMRKLOC,RM22:=R22FROMRKLOC,
RM16:=R16FROMRKLOC,RM26:=R26FROMRKLOC,RM51:=R51FROMRKLOC,RM52:=R52FROMRKLOC,
RM33:=R33FROMRKLOC,RM34:=R34FROMRKLOC,RM43:=R43FROMRKLOC,RM44:=R44FROMRKLOC,
RM55:=1.,RM66:=1.;

use,sequence=seqname;
twiss;

betxFROMRKLOC   = table(twiss, markFROMRKLOC, betx);
betyFROMRKLOC   = table(twiss, markFROMRKLOC, bety);
alfxFROMRKLOC   = table(twiss, markFROMRKLOC, alfx);
alfyFROMRKLOC   = table(twiss, markFROMRKLOC, alfy);
dispxFROMRKLOC  = table(twiss, markFROMRKLOC, dx);
disppxFROMRKLOC = table(twiss, markFROMRKLOC, dpx);
R11FROMRKLOC   := cos(twopidphix) + alfxFROMRKLOC * sin(twopidphix);
R12FROMRKLOC   := betxFROMRKLOC * sin(twopidphix);
R22FROMRKLOC   := cos(twopidphix) - alfxFROMRKLOC * sin(twopidphix);
R21FROMRKLOC   :=-sin(twopidphix) * (1 + alfxFROMRKLOC^2) / betxFROMRKLOC;
R33FROMRKLOC   := cos(twopidphiy) + alfyFROMRKLOC * sin(twopidphiy);
R34FROMRKLOC   := betyFROMRKLOC * sin(twopidphiy);
R44FROMRKLOC   := cos(twopidphiy) - alfyFROMRKLOC * sin(twopidphiy);
R43FROMRKLOC   :=-sin(twopidphiy) * (1 + alfyFROMRKLOC^2) / betyFROMRKLOC;
//R16FROMRKLOC   :=  dispxFROMRKLOC * (1 - R11FROMRKLOC) - R12FROMRKLOC * disppxFROMRKLOC;
//R26FROMRKLOC   := disppxFROMRKLOC * (1 - R22FROMRKLOC) - R21FROMRKLOC * dispxFROMRKLOC;
//R51FROMRKLOC   := R21FROMRKLOC * R16FROMRKLOC - R11FROMRKLOC * R26FROMRKLOC;
//R52FROMRKLOC   := R22FROMRKLOC * R16FROMRKLOC - R12FROMRKLOC * R26FROMRKLOC;

show,twopidphix,alfxFROMRKLOC, betxFROMRKLOC, R12FROMRKLOC;


seqedit,sequence=seqname;
install, element=MATFROMRKLOC,at=0.0,from=markFROMRKLOC;
endedit;
use, sequence=seqname;
};
""")

    LOC = 0.0001
    FROMRK = 'matmark'
    mad.input(f"EXEC PHASESHIFT(toy,{LOC},{FROMRK},twopidphix,twopidphiy);")

twiss = mad.twiss()
summ  = mad.table.summ

if use_mat:
    print(summ['q1'][0]-summ0['q1'][0],dqx_mat)
    print(summ['q2'][0]-summ0['q2'][0],dqy_mat)
    if False:
        plt.figure(0)
        plt.plot(twiss0['s'],twiss0['betx'],'-b')
        plt.plot(twiss['s'],twiss['betx'],'--k')
        plt.plot(twiss0['s'],twiss0['bety'],'-g')
        plt.plot(twiss['s'],twiss['bety'],'--k')

    for element in mad.sequence['toy'].elements:
        if element.name == f'matmatmark{LOC}':
            mat_name = element.name
            print('madx',element)

if True:
    mad.input('''
option,trace;
track,dump;
start,x=1e-6,y=0.0;
start,x=0.0,y=1e-6; 
dynap,fastune,turns=1024;
endtrack;''')
    thintrack_qx = mad.table.dynaptune.tunx[0]
    thintrack_qy = mad.table.dynaptune.tuny[1]

# Build Xtrack line importing MAD-X expressions
line = xt.Line.from_madx_sequence(mad.sequence['toy'],
                                  deferred_expressions=False)

for element in mad.sequence['toy'].elements:
    print('mad:',element)
for element in line.elements:
    print('xt',element,element.length)

if use_mat:
    mat_element = line.element_dict[mat_name]
    print(mat_element.m1)
    mat_element.radiation_flag = 2

# Define reference particle
line.particle_ref = xp.Particles(mass0=xp.ELECTRON_MASS_EV, q0=+1,
                                 gamma0=mad.sequence['toy'].beam.gamma)

particles = xp.Particles(mass0=xp.ELECTRON_MASS_EV,
                         q0=+1,
                         gamma0=mad.sequence['toy'].beam.gamma,
                         x = [1E-6,0.0],
                         y = [0.0,1E-6])
particles._init_random_number_generator()

if True:
    num_turns = 1024
    x = np.zeros(num_turns)
    px = np.zeros(num_turns)
    y = np.zeros(num_turns)
    py = np.zeros(num_turns)
    zeta = np.zeros(num_turns)
    delta = np.zeros(num_turns)
    turns = np.arange(num_turns)
    for turn in turns:
        mat_element.track(particles)
        x[turn] = particles.x[0]
        px[turn] = particles.px[0]
        y[turn] = particles.y[1]
        py[turn] = particles.py[1]
        zeta[turn] = particles.zeta[0]
        delta[turn] = particles.py[0]

    harmon = HarmonicAnalysis(x)
    freqs,coefs = harmon.laskar_method(1)
    print('Qx',freqs[0],dqx_mat)
    harmon = HarmonicAnalysis(y)
    freqs,coefs = harmon.laskar_method(1)
    print('Qy',freqs[0],dqy_mat)
    turns = np.arange(num_turns)
    plt.figure(10)
    plt.plot(x,px,'xb')
    plt.figure(20)
    plt.plot(y,py,'xb')
    plt.figure(30)
    plt.plot(zeta,delta,'xb')


#particles0 = particles.copy()
#mat_element.track(particles)
#print('x',particles.x[0]-particles0.x[0])
#print('px',particles.px[0]-particles0.px[0])
#print('y',particles.y[0]-particles0.y[0])
#print('py',particles.py[0]-particles0.py[0])
#print('zeta',particles.zeta[0]-particles0.zeta[0])
#print('delta',particles.delta[0]-particles0.delta[0])
#exit()

# Build tracker
num_turns = 1024
tracker = xt.Tracker(line=line)
tracker.track(particles, num_turns=num_turns, turn_by_turn_monitor=True)

mon = tracker.record_last_track
harmon = HarmonicAnalysis(mon.x[0,:])
freqs,coefs = harmon.laskar_method(1)
print('Qx',freqs[0],summ['q1'][0],summ0['q1'][0],thintrack_qx)
print('Qx diff',freqs[0]-summ['q1'][0]+np.floor(summ['q1'][0]),thintrack_qx-summ['q1'][0]+np.floor(summ['q1'][0]))
harmon = HarmonicAnalysis(mon.y[1,:])
freqs,coefs = harmon.laskar_method(1)
print('Qy',freqs[0],summ['q2'][0],summ0['q2'][0],thintrack_qy)
print('Qy diff',freqs[0]-summ['q2'][0]+np.floor(summ['q2'][0]),thintrack_qy-summ['q2'][0]+np.floor(summ['q2'][0]))
turns = np.arange(num_turns)
plt.figure(1)
plt.plot(mon.x[0,:],mon.px[0,:],'xb')
plt.plot(mon.x[1,:],mon.px[1,:],'xg')
plt.figure(2)
plt.plot(mon.y[0,:],mon.py[0,:],'xb')
plt.plot(mon.y[1,:],mon.py[1,:],'xg')
plt.figure(3)
plt.plot(mon.zeta[0,:],mon.delta[0,:],'xb')
plt.plot(mon.zeta[1,:],mon.delta[1,:],'xg')
plt.show()

