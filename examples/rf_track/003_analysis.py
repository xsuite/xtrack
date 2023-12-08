import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

xt    = loadmat('particles_xt.mat')
rft_v = loadmat('particles_rft_volume.mat')
rft_s = loadmat('particles_rft_sbend.mat')

X = xt['x']
V = rft_v['v']
S = rft_s['s']

plt.figure(1)
plt.scatter(X[:,0]*1e3, X[:,1]*1e3, label='Xtrack',      s=100, facecolors='none', edgecolors='b')
plt.scatter(V[:,0]*1e3, V[:,1]*1e3, label='RFT::Volume', s=10, facecolors='m', edgecolors='none')
plt.scatter(S[:,0]*1e3, S[:,1]*1e3, label='RFT::SBend',  s=10, facecolors='c', edgecolors='none')
plt.xlabel("$x$ [mm]")
plt.ylabel("$p_x$ [mrad]")
plt.legend()
plt.show()

plt.figure(2)
plt.scatter(X[:,2]*1e3, X[:,3]*1e3, label='Xtrack',      s=100, facecolors='none', edgecolors='b')
plt.scatter(V[:,2]*1e3, V[:,3]*1e3, label='RFT::Volume', s=10, facecolors='m', edgecolors='none')
plt.scatter(S[:,2]*1e3, S[:,3]*1e3, label='RFT::SBend',  s=10, facecolors='c', edgecolors='none')
plt.xlabel("$y$ [mm]")
plt.ylabel("$p_y$ [mrad]")
plt.legend()
plt.show()

plt.figure(3)
plt.scatter(X[:,4]*1e3, X[:,5]*1e3, label='Xtrack',      s=100, facecolors='none', edgecolors='b')
plt.scatter(V[:,4]*1e3, V[:,5]*1e3, label='RFT::Volume', s=10, facecolors='m', edgecolors='none')
plt.scatter(S[:,4]*1e3, S[:,5]*1e3, label='RFT::SBend',  s=10, facecolors='c', edgecolors='none')
plt.xlabel("$t$ [mm/c]")
plt.ylabel("$p_t$ [mrad]")
plt.legend()
plt.show()
