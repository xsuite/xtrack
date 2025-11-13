import numpy as np
from cpymad.madx import Madx

import xtrack as xt

import matplotlib.pyplot as plt

mad1 = Madx()

# Load mad model and apply element shifts
mad1.input('''
call, file = '../../test_data/psb_chicane/psb.seq';
call, file = '../../test_data/psb_chicane/psb_fb_lhc.str';

beam, particle=PROTON, pc=0.5708301551893517;
use, sequence=psb1;

select,flag=error,clear;
select,flag=error,pattern=bi1.bsw1l1.1;
ealign, dx=-0.0057;

select,flag=error,clear;
select,flag=error,pattern=bi1.bsw1l1.2;
select,flag=error,pattern=bi1.bsw1l1.3;
select,flag=error,pattern=bi1.bsw1l1.4;
ealign, dx=-0.0442;

!twiss;
''')

print(f'{mad1.sequence.psb1.expanded_elements["bi1.bsw1l1.1"].align_errors.dx=}')
print(f'{mad1.sequence.psb1.expanded_elements["bi1.bsw1l1.2"].align_errors.dx=}')
print(f'{mad1.sequence.psb1.expanded_elements["bi1.bsw1l1.3"].align_errors.dx=}')
print(f'{mad1.sequence.psb1.expanded_elements["bi1.bsw1l1.4"].align_errors.dx=}')
"""
mad2 = Madx()
# Load mad model and apply element shifts
mad2.input('''
call, file = '../../test_data/psb_chicane/psb.seq';
call, file = '../../test_data/psb_chicane/psb_fb_lhc.str';

beam, particle=PROTON, pc=0.5708301551893517;
use, sequence=psb1;

select,flag=error,clear;
select,flag=error,pattern=bi1.bsw1l1.1.*;
ealign, dx=-0.0057;

select,flag=error,clear;
select,flag=error,pattern=bi1.bsw1l1.2.*;
select,flag=error,pattern=bi1.bsw1l1.3.*;
select,flag=error,pattern=bi1.bsw1l1.4.*;
ealign, dx=-0.0442;

twiss;
''')

print(f'{mad2.sequence.psb1.expanded_elements["bi1.bsw1l1.1"].align_errors.dx=}')
print(f'{mad2.sequence.psb1.expanded_elements["bi1.bsw1l1.2"].align_errors.dx=}')
print(f'{mad2.sequence.psb1.expanded_elements["bi1.bsw1l1.3"].align_errors.dx=}')
print(f'{mad2.sequence.psb1.expanded_elements["bi1.bsw1l1.4"].align_errors.dx=}')
"""