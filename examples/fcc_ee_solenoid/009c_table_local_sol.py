import xtrack as xt
import numpy as np

r0 = 0.13
ds_start = 1.4
ds_end = 2.29
B0 = 3.
sol_half_length = 1.3
r0_local = 0.13

B0_local_list = np.array([0.0001, 0.2, 0.4, 0.6, 0.8,
                          1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
                          2.2, 2.4, 2.6, 2.8, 3.0])
l_local_list = np.array([0.4, 0.6, 0.8])

# B0_local_list = np.array([0.0001, 0.2, 0.4, 0.6, 0.8,
#                           1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
#                           2.2, 2.4, 2.6, 2.8, 3.0,
#                           3.2, 3.4, 3.6])
# l_local_list = np.array([0.6])

cols = {str(ll): 0*B0_local_list for ll in l_local_list}
cols['B0_local'] = np.array(list(map(str, B0_local_list)))

tt = xt.Table(cols, index='B0_local')

for B0_local in B0_local_list:
    for ll in l_local_list:
        fname = (f'B0_{B0:.3f}_r0_{r0:.3f}_sol_half_length_{sol_half_length:.3f}'
                f'_ds_start_{ds_start:.3f}_ds_end_{ds_end:.3f}')
        if B0_local !=0:
            fname += f'_B0_local_{B0_local:.3f}_r0_local_{r0_local:.3f}_l_local_{ll:.3f}'
        fpath = 'results/' + fname + '.json'

        ddd = xt.json.load(fpath)
        tt[str(ll), str(B0_local)] = ddd['gemitt_y']
print()
print(f'{r0=:.3f} {ds_start=:.3f}')
print()
tt.show(digits=3)

import matplotlib.pyplot as plt
plt.close('all')

# One curve
l_plot = 0.6
plt.figure(1)
plt.plot(B0_local_list, tt[str(l_plot)]*1e12)
plt.xlabel(r'$B_\text{local}$ [T]')
plt.ylabel('Vertical emittance [pm]')
plt.suptitle(f'l_local = {l_plot}')
plt.ylim(0.4, None)
plt.show()

# All curves
plt.figure(2)
for ll in l_local_list:
    plt.plot(B0_local_list, tt[str(ll)]*1e12, label=f'l_local = {ll}')
plt.xlabel(r'$B_\text{local}$ [T]')
plt.ylabel('Vertical emittance [pm]')
plt.ylim(0.4, None)
plt.legend()

plt.show()
