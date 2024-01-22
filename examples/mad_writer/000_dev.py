from cpymad.madx import Madx
import xtrack as xt

mad = Madx()

# Element definitions
mad.input("""

a = 1.;
b := sin(3*a) + cos(2*a);

cav1: rfcavity, freq:=a*10, lag:=a*0.5, volt:=a*6;
testseq: sequence, l=10;
c1: cav1, at=0.2, apertype=circle, aperture=0.01;
endsequence;
"""
)

# Beam
mad.input("""
beam, particle=proton, gamma=1.05, sequence=testseq;
""")

mad.use('testseq')

seq = mad.sequence['testseq']

line = xt.Line.from_madx_sequence(sequence=seq, deferred_expressions=True)


expr_str = str(line.vars['b']._expr)

fff = line._var_management['data']['functions']
for nn in fff._mathfunctions:
    expr_str = expr_str.replace(f'f.{nn}(', f'{nn}(')

expr_str = expr_str.replace("'", "")
expr_str = expr_str.replace('"', "")

# transform vars[...] in (...)
while "vars[" in expr_str:
    before, after = tuple(*[expr_str.split("vars[", 1)])
    # find the corresponding closing bracket
    count = 1
    for ii, cc in enumerate(after):
        if cc == "]":
            count -= 1
        elif cc == "[":
            count += 1
        if count == 0:
            break

    expr_str = before + "(" + after[:ii] + ")" + after[ii+1:]


