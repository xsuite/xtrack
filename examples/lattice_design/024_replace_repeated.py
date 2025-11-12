import xtrack as xt

env = xt.Environment()
line = env.new_line(length=10.0,
    components=[
        env.new('m', xt.Marker, at=5.0)])

line.get_table().cols['element_type']
# Table: 4 rows, 2 cols
# name         element_type
# ||drift_1::0 Drift
# m            Marker
# ||drift_1::1 Drift
# _end_point

breakpoint()
line.replace_all_repeated_elements()

line.get_table().cols['element_type']
# Table: 4 rows, 2 cols
# name        element_type
# ||drift_1.0 Drift
# m           Marker
# ||drift_1.1 Drift
# _end_point