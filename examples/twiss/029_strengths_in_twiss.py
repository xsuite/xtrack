
from cpymad.madx import Madx
import xtrack as xt

test_data_folder = '../../test_data'

test_data_folder_str = str(test_data_folder)

mad1=Madx(stdout=False)
mad1.call(test_data_folder_str + '/hllhc15_thick/lhc.seq')
mad1.call(test_data_folder_str + '/hllhc15_thick/hllhc_sequence.madx')
mad1.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
mad1.use('lhcb1')
mad1.input('beam, sequence=lhcb2, particle=proton, energy=7000, bv=-1;')
mad1.use('lhcb2')
mad1.call(test_data_folder_str + '/hllhc15_thick/opt_round_150_1500.madx')
mad1.twiss()

collider = xt.Multiline.from_madx(madx=mad1)
tw = collider.twiss(strengths=True, method='4d')