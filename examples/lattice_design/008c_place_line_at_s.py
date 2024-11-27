import xtrack as xt
import numpy as np

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=2e9, mass0=xt.PROTON_MASS_EV)

sub_line = env.new_line(components=[
    env.new('mid_subline', xt.Marker, at=1.0),
    env.new('mq1_subline', xt.Quadrupole, length=0.3, at=-0.5, from_='mid_subline'),
    env.new('mq2_subline', xt.Quadrupole, length=0.3, at=0.5, from_='mid_subline')])

line = env.new_line(components=[
    env.new('bb', xt.Bend, length=1.0, at=2),
    env.place(sub_line, at=5.0),
    env.place(sub_line, at=8.0),
    ])

line.get_table()
# is:
# Table: 17 rows, 8 cols
# name                       s element_type isthick isreplica parent_name ...
# drift_4                    0 Drift           True     False None
# bb                       1.5 Bend            True     False None
# drift_5                  2.5 Drift           True     False None
# drift_1::0             4.175 Drift           True     False None
# mq1_subline::0         4.525 Quadrupole      True     False None
# drift_2::0             4.825 Drift           True     False None
# mid_subline::0         5.175 Marker         False     False None
# drift_3::0             5.175 Drift           True     False None
# mq2_subline::0         5.525 Quadrupole      True     False None
# drift_6                5.825 Drift           True     False None
# drift_1::1             7.175 Drift           True     False None
# mq1_subline::1         7.525 Quadrupole      True     False None
# drift_2::1             7.825 Drift           True     False None
# mid_subline::1         8.175 Marker         False     False None
# drift_3::1             8.175 Drift           True     False None
# mq2_subline::1         8.525 Quadrupole      True     False None
# _end_point             8.825                False     False None
