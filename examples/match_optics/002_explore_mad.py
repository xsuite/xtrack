from cpymad.madx import Madx
import xtrack as xt

def get_beta_blocks(mad, names):
    out = {}
    for nn in names:
        dd = {}
        for kk in list(mad.beta0.keys()):
            mad.input(f'tttt = {nn}->{kk};')
            dd[kk] = mad.globals['tttt']
        out[nn] = dd
    return out

mad=Madx()
mad.call('../../test_data/hllhc15_thick/lhc.seq')
mad.call('../../test_data/hllhc15_thick/hllhc_sequence.madx')
mad.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
mad.use('lhcb1')
mad.input('beam, sequence=lhcb2, particle=proton, energy=7000;')
mad.use('lhcb2')
mad.call("../../test_data/hllhc15_thick/opt_round_150_1500.madx")

mad.call('../../../hllhc15/toolkit/macro.madx')

# # Example of "select" command
# mad.input('exec,select(3,23,34,b1);')
# out_sel_ir3= get_beta_blocks(mad, ['bir3b1', 'eir3b1'])

# # Example of "selectIRAUX" command
mad.input(
    '''
    scxir1=betx_IP1/betx0_IP1;
    scyir1=bety_IP1/bety0_IP1;
    scxir5=betx_IP5/betx0_IP5;
    scyir5=bety_IP5/bety0_IP5;
    exec,selectIRAUX(3,4,5,6,7,b1,scxir5,scyir5,betx0_ip5,bety0_ip5);
    '''
    )
out_sel_aux_5 = get_beta_blocks(mad, [
    'bir4b1', 'eir4b1', 'bir5b1', 'eir5b1', 'bir6b1', 'eir6b1'])

# Example of "selectIR15" command
mad.input('''
    scxir1=betx_IP1/betx0_IP1;
    scyir1=bety_IP1/bety0_IP1;
    scxir5=betx_IP5/betx0_IP5;
    scyir5=bety_IP5/bety0_IP5;

    value,scxir1,scyir1,scxir5,scyir5;

    exec,selectIR15(5,45,56,b1);
    ''')
out_sel_ir15_5 = get_beta_blocks(mad, ['bir5b1', 'eir5b1'])

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()
collider.vars.load_madx_optics_file(
    "../../test_data/hllhc15_thick/opt_round_150_1500.madx")

twpresq_r = collider.lhcb1.twiss(
    twiss_init=xt.TwissInit(
        element_name='ip5', betx=0.5, bety=0.5, line=collider.lhcb1),
    ele_start='ip5', ele_stop='ip6')
twpresq_l = collider.lhcb1.twiss(
    twiss_init=xt.TwissInit(
        element_name='ip5', betx=0.5, bety=0.5, line=collider.lhcb1),
    ele_start='ip4', ele_stop='ip5')

twsq_r = collider.lhcb1.twiss(
    twiss_init=xt.TwissInit(
        element_name='ip5', betx=0.15, bety=0.15, line=collider.lhcb1),
    ele_start='ip5', ele_stop='ip6')
twsq_l = collider.lhcb1.twiss(
    twiss_init=xt.TwissInit(
        element_name='ip5', betx=0.15, bety=0.15, line=collider.lhcb1),
    ele_start='ip4', ele_stop='ip5')

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(twpresq_r.s, twpresq_r.betx, 'g', label='presq')
plt.plot(twpresq_l.s, twpresq_l.betx, 'g')
plt.plot(twsq_r.s, twsq_r.betx, 'r', label='sq')
plt.plot(twsq_l.s, twsq_l.betx, 'r')

plt.plot(twpresq_r['s', 'e.ds.r5.b1'], out_sel_aux_5['eir5b1']['betx'], 'or', label='sel_aux')
plt.plot(twpresq_r['s', 'e.ds.r5.b1'], out_sel_ir15_5['eir5b1']['betx'], 'og', label='sel_15')
plt.legend()
plt.show()
