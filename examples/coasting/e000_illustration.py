import numpy as np
import matplotlib.pyplot as plt

v = 1.
l_ref = 1

t_ref = l_ref/v

l = np.array([0.95, 1, 1.05])

t_max = 35
t = np.linspace(0, t_max, 10000)

s = np.zeros((len(l), len(t)))
for il, ll in enumerate(l):
    s[il, :] = np.mod(l_ref/2 + v * t * ll/l_ref, l_ref)

passing = np.diff(s, prepend=0)< 0

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1)
sp1 = plt.subplot(311)
sp2 = plt.subplot(312, sharex=sp1)
sp3 = plt.subplot(313, sharex=sp1)

sp1.plot(t, passing[0, :])
sp2.plot(t, passing[1, :])
sp3.plot(t, passing[2, :])

t_turn = np.arange(0, t_max, t_ref)
for tt in t_turn:
    for sp in [sp1, sp2, sp3]:
        sp.axvline(tt, color='k', linestyle='--', alpha=0.4)

plt.show()

