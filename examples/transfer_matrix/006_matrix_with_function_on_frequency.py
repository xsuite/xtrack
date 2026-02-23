import xtrack as xt

lmap = xt.LineSegmentMap(
                        length=628.,
                        qx=6.23,
                        qy=6.41,
                        momentum_compaction_factor=1/2.3**2,
                        betx=1.,
                        bety=1.,
                        longitudinal_mode          = 'nonlinear',
                        voltage_rf                 = [1e4],
                        frequency_rf               = [20e6],
                        lag_rf                     = [0.0],
                    )

env = xt.Environment()
env.elements['lmap'] = lmap

line = env.new_line(components=['lmap'])

n_cav = 3
t_rf = [0, 1e-3, 2e-3]
v_rf = [20e3, 21e3, 20e3] # in V

line.functions['fun_v_rf'] = xt.FunctionPieceWiseLinear(x=t_rf, y=v_rf)
voltage = line.functions['fun_v_rf'](line.ref['t_turn_s'])

line['lmap'].voltage_rf[0] = voltage / n_cav

line.ref['lmap'].voltage_rf[0]._info()
# prints:
# #  element_refs['lmap'].voltage_rf[0]._get_value()
#    element_refs['lmap'].voltage_rf[0] = 6666.666666666667

# #  element_refs['lmap'].voltage_rf[0]._expr
#    element_refs['lmap'].voltage_rf[0] = (f['fun_v_rf'](vars['t_turn_s']) / 3)

# #  element_refs['lmap'].voltage_rf[0]._expr._get_dependencies()
#    vars['t_turn_s'] = 0.0
#    f['fun_v_rf'] = <xdeps.functions.FunctionPieceWiseLinear object at 0x12315fc50>

# #  element_refs['lmap'].voltage_rf[0] does not influence any target