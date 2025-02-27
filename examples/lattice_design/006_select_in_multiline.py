import xtrack as xt

# --- Parameters
seq         = 'lhcb1'
ip_name     = 'ip1'
s_marker    = f'e.ds.l{ip_name[-1]}.b1'
e_marker    = f's.ds.r{ip_name[-1]}.b1'
#-------------------------------------


collider_file = '../../test_data/hllhc15_collider/collider_00_from_mad.json'


# Load the machine and select line
collider= xt.Environment.from_json(collider_file)
collider.vars['test_vars'] = 3.1416
line   = collider[seq]
line_sel    = line.select(s_marker,e_marker)

assert line_sel.element_dict is line.element_dict
assert line.get('ip1') is line_sel.get('ip1')

line_sel['aaa'] = 1e-6
assert line_sel['aaa'] == 1e-6
assert line['aaa'] == 1e-6

line_sel.ref['mcbch.7r1.b1'].knl[0] += line.ref['aaa']
assert (str(line.ref['mcbch.7r1.b1'].knl[0]._expr)
        == "((-vars['acbch7.r1b1']) + vars['aaa'])")
assert (str(line_sel.ref['mcbch.7r1.b1'].knl[0]._expr)
        == "((-vars['acbch7.r1b1']) + vars['aaa'])")
assert line_sel.get('mcbch.7r1.b1').knl[0] == 1e-6
assert line.get('mcbch.7r1.b1').knl[0] == 1e-6

