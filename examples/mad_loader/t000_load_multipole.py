import xtrack as xt

mad_src = '''
    m1: multipole, l=1.0, knl={0, 0.01, 0.0, 0,0, 0.1, 0.3, 0.7};

    seq: sequence, l=10.0;
    m1a: m1, at=2.0;
    endsequence;
'''

env = xt.load(string=mad_src, format='madx')