import xtrack as xt
import numpy as np

r0 = 0.13
ds_start = 1.5
ds_end = 2.29

B0_list = np.array([2, 2.5, 2.75, 3])
s_screen_sol_list = np.array([1.2, 1.3, 1.4, 1.5])

cols = {str(ss): 0*B0_list for ss in s_screen_sol_list}
cols['B'] = np.array(list(map(str, B0_list)))

tt = xt.Table(cols, index='B')

for B0 in B0_list:
    for ss in s_screen_sol_list:
        fname = (f'B0_{B0:.3f}_r0_{r0:.3f}_sol_half_length_{ss:.3f}'
                f'_ds_start_{ds_start:.3f}_ds_end_{ds_end:.3f}.json')
        ddd = xt.json.load('results/'+fname)
        tt[str(ss), str(B0)] = ddd['gemitt_y']
print()
print(f'{r0=:.3f} {ds_start=:.3f}')
print()
tt.show(digits=3)
