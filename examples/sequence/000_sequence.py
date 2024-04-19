import numpy as np
from xtrack import Line, Node, Multipole

# Or from a sequence definition:
elements = {
    'quad': Multipole(length=0.3, knl=[0, +0.50]),
    'bend': Multipole(length=0.5, knl=[np.pi / 12], hxl=[np.pi / 12]),
}
sequences = {
    'arc': [Node(1.0, 'quad'), Node(4.0, 'bend', from_='quad')],
}
line = Line.from_sequence([
        Node( 0.0, 'arc'),
        Node(10.0, 'arc', name='section2'),
        Node( 3.0, Multipole(knl=[0, 0, 0.1]), from_='section2', name='sext'),
        Node( 3.0, 'quad', name='quad_5', from_='sext'),
    ], length=20,
    elements=elements, sequences=sequences,
    auto_reorder=True, copy_elements=False,
)

line.get_table().show()
# prints:
# #
# name          s element_type isthick isreplica parent_name iscollective
# arc           0 Marker         False     False        None        False
# drift         0 Drift           True     False        None        False
# arcquad       1 Multipole      False     False        None        False
# drift1        1 Drift           True     False        None        False
# arcbend       5 Multipole      False     False        None        False
# drift2        5 Drift           True     False        None        False
# section2     10 Marker         False     False        None        False
# drift3       10 Drift           True     False        None        False
# section2quad 11 Multipole      False     False        None        False
# drift4       11 Drift           True     False        None        False
# sext         13 Multipole      False     False        None        False
# drift5       13 Drift           True     False        None        False
# section2bend 15 Multipole      False     False        None        False
# drift6       15 Drift           True     False        None        False
# quad_5       16 Multipole      False     False        None        False
# drift7       16 Drift           True     False        None        False
# _end_point   20                False     False        None        False