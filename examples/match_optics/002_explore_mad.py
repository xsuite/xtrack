from cpymad.madx import Madx

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
