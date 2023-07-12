import numpy as np
import xtrack as xt

import lhc_match as lm


collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()
collider.vars.load_madx_optics_file('../../../hllhc15/toolkit/macro.madx')

arc_periodic_solution =lm.get_arc_periodic_solution(collider)