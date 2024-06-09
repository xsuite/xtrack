# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.build_trackers()

collider.lhcb1.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['reverse'] = True

line = collider.lhcb2

ele_ref = 'ip8'

# For the closed twiss we can do this
tw = line.twiss()
t1 = tw.rows[ele_ref:]
t2 = tw.rows[:ele_ref]
tw_ip8 = xt.TwissTable.concatenate([t1, t2])

t1o = line.twiss(start=ele_ref, end=xt.END, betx=1.5, bety=1.5)
init_part2 = t1o.get_twiss_init('_end_point')

# Dummy twiss to get the name at the start of the secon part
init_part2.element_name = line.twiss(start=xt.START, end=xt.START, betx=1, bety=1).name[0]

t2o = line.twiss(start=xt.START, end=ele_ref, init=init_part2)
# remove repeated element
t2o = t2o.rows[:-1]
t2o.name[-1] = '_end_point'
two_ip8 = xt.TwissTable.concatenate([t1o, t2o])
