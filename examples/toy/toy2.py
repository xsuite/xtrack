import numpy as np
from matplotlib import pyplot as plt
from cpymad.madx import Madx

import xtrack as xt
import xpart as xp

m1 = np.reshape(np.random.randn(36),(6,6))
mad = Madx()
mad.option(echo=True, debug=False, info=False, warn=False)
x0 = np.random.randn()
px0 = np.random.randn()
y0 = np.random.randn()
py0 = np.random.randn()
mad.input(f'''
beam,particle=POSITRON;
mat:matrix,l=1,rm11={m1[0,0]},rm12={m1[0,1]},rm13={m1[0,2]},rm14={m1[0,3]},rm15={m1[0,4]},rm16={m1[0,5]},rm21={m1[1,0]},rm22={m1[1,1]},rm23={m1[1,2]},rm24={m1[1,3]},rm25={m1[1,4]},rm26={m1[1,5]},rm31={m1[2,0]},rm32={m1[2,1]},rm33={m1[2,2]},rm34={m1[2,3]},rm35={m1[2,4]},rm36={m1[2,5]},rm41={m1[3,0]},rm42={m1[3,1]},rm43={m1[3,2]},rm44={m1[3,3]},rm45={m1[3,4]},rm46={m1[3,5]},rm51={m1[4,0]},rm52={m1[4,1]},rm53={m1[4,2]},rm54={m1[4,3]},rm55={m1[4,4]},rm56={m1[4,5]},rm61={m1[5,0]},rm62={m1[5,1]},rm63={m1[5,2]},rm64={m1[5,3]},rm65={m1[5,4]},rm66={m1[5,5]};
matseq:sequence,l=1.0;
mat1:mat, at=0.5;
endsequence;
use,sequence=matseq;
track,onepass;
start,x={x0},px={px0},y={y0},py={py0};
run,turns=1;
endtrack;
''')
#print(mad.sequence['matseq'].elements['mat1'].rm12)
line = xt.Line.from_madx_sequence(mad.sequence['matseq'])
for element in mad.sequence['matseq'].elements:
    print('mad:',element)
for element in line.elements:
    print('xt',element,element.length)
particles = xp.Particles(mass0=xp.ELECTRON_MASS_EV,q0=+1,gamma0=mad.sequence['matseq'].beam.gamma,x = x0,px=px0,y=y0,py=py0)
particles0 = particles.copy()
#mat = line.elements[1]
#mat.track(particles)
tracker = xt.Tracker(line=line)
tracker.track(particles, num_turns=1)
print('thintrack',mad.table.tracksumm.x)
print('xt',particles0.x,particles.x)
print('thintrack',mad.table.tracksumm.px)
print('xt',particles0.px,particles.px)
print('thintrack',mad.table.tracksumm.y)
print('xt',particles0.y,particles.y)
print('thintrack',mad.table.tracksumm.py)
print('xt',particles0.py,particles.py)

print('-------')
print('x',particles0.x-mad.table.tracksumm.x[0],particles.x-mad.table.tracksumm.x[1])
print('px',particles0.px-mad.table.tracksumm.px[0],particles.px-mad.table.tracksumm.px[1])
print('y',particles0.y-mad.table.tracksumm.y[0],particles.y-mad.table.tracksumm.y[1])
print('py',particles0.py-mad.table.tracksumm.py[0],particles.py-mad.table.tracksumm.py[1])
