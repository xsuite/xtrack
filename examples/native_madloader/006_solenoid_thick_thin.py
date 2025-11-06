import xtrack as xt

mad_str = """
    sol1: solenoid, l=1.0, ks=0.5;
    sol2: sol1, lrad=0.2;
    sol3: sol2;
"""

env = xt.load(string=mad_str, format='madx')