import xtrack as xt

elements = {
    'd1': xt.Drift(length=1),
    'm1': xt.Marker(),
    'd2': xt.Drift(length=1),
}

line=xt.Line(elements=elements,
             element_names=list(elements.keys()))

line.insert_element(element=xt.Bend(length=1.), name='m1', at_s=0.5)