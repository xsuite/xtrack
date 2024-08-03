import numpy as np
import matplotlib.pyplot as plt

v = 1.
l_ref = 1

t_ref = l_ref/v

l = np.array([0.95, 1, 1.05])

t_max = 80
t = np.linspace(0, t_max, 10000)

s = np.zeros((len(l), len(t)))
for il, ll in enumerate(l):
    s[il, :] = np.mod(0.62*l_ref + v * t * ll/l_ref, l_ref)

passing = np.diff(s, prepend=0) < 0

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1, figsize=(6.4*1.8*0.8, 4.8*0.8))
sp1 = plt.subplot(311)
sp2 = plt.subplot(312, sharex=sp1)
sp3 = plt.subplot(313, sharex=sp1)

sp1.stem(t, passing[1, :], markerfmt=' ', basefmt='C0', linefmt='C0')
sp2.stem(t, passing[0, :], markerfmt=' ', basefmt='C1', linefmt='C1')
sp3.stem(t, passing[2, :], markerfmt=' ', basefmt='C2', linefmt='C2')

t_turn = np.arange(0, t_max, t_ref)
for tt in t_turn:
    for sp in [sp1, sp2, sp3]:
        sp.axvline(tt, color='k', linestyle='--', alpha=0.4)

sp3.set_xlim(0, t_max/t_ref)
sp3.set_xlabel(r'$t~/~T_0$')

for sp in [sp1, sp2, sp3]:
    sp.set_ylim(0, 1.1)

plt.subplots_adjust(bottom=.14, top=.95, hspace=0.3)

# For zoom
# plt.subplots_adjust(right=.6)

plt.show()

