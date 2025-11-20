import xtrack as xt
from xtrack._temp.python_lattice_writer import lattice_py_generation as lpg

from xtrack.mad_parser.loader import CONSTANTS

fpath = '../../test_data/lhc_2024/lhc.seq'

with open(fpath, 'r') as fid:
    seq_text = fid.read()

assert ' at=' in seq_text
assert ',at=' not in seq_text
assert 'at =' not in seq_text
seq_text = seq_text.replace(' at=', 'at:=') # to have the expressions in the at

env = xt.load(string=seq_text, format='madx', reverse_lines=['lhcb2'])

# Force k0_from_h to False (they are all provided)
for nn in list(env.elements.keys()):
    if hasattr(env.elements[nn], 'k0_from_h'):
        env.elements[nn].k0_from_h = False

lpg.write_py_lattice_file(env, output_fname='lhc_seq.py')
