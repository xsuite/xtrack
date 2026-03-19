from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
import numpy as np

sf = SolenoidField(L=3., a=0.2, B0=2., z0=0)

theta = -0.015
z = np.linspace(-3, 3, 1001)
s = z / np.cos(theta)
x = s * np.sin(theta)
y = 0 * x

bx, by, bz = sf.get_field(x, y, z)


import matplotlib.pyplot as plt
plt.close('all')
plt.plot(s, bx, label='bx')
plt.plot(s, by, label='by')
plt.plot(s, bz, label='bz')
plt.legend()
plt.xlabel('s [m]')
plt.ylabel('B [T]')
plt.title('Field on the reference trajectory')

plt.show()
