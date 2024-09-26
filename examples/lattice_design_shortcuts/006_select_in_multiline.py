import xtrack as xt

# --- Parameters
seq         = 'lhcb1'
ip_name     = 'ip1'
s_marker    = f'e.ds.l{ip_name[-1]}.b1'
e_marker    = f's.ds.r{ip_name[-1]}.b1'
#-------------------------------------


collider_file = '../../test_data/hllhc15_collider/collider_00_from_mad.json'


# Load the machine and select line
collider= xt.Multiline.from_json(collider_file)
collider.vars['test_vars'] = 3.1416
line0   = collider[seq]
line    = line0.select(s_marker,e_marker)

line.env._var_management = None
line._var_management = None
line.env._in_multiline = collider
line.env._name_in_multiline = line0._name_in_multiline
line._in_multiline = collider
line._name_in_multiline = line0._name_in_multiline

print('collider and line0 share vars:',line0.vars is collider.vars)
print('selected line and line0 share vars:',line.vars is line0.vars)
print('test vars in line0:', line0.vars['test_vars']._value)
print('test vars in selected line:', line.vars['test_vars']._value)