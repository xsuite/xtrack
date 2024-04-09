import xtrack as xt

bend = xt.Bend(k0=0.4, h=0.3, length=1)

line = xt.Line(elements=[bend, xt.Replica(_parent_name='e0')])

line.build_tracker()

tt = line.get_table(attr=True)
tt.show()