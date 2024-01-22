from cpymad.madx import Madx
import xtrack as xt
import xdeps as xd

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


def expr_to_mad_str(expr):

    expr_str = str(expr)

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

    return expr_str

def mad_str_or_value(var):
    vv = _get_expr(var)
    if _is_ref(vv):
        return expr_to_mad_str(vv)
    else:
        return vv

def mad_assignment(lhs, rhs):
    if _is_ref(rhs):
        return f"{lhs} := {mad_str_or_value(rhs)}"
    if isinstance(rhs, str):
        return f"{lhs} := {rhs}"
    else:
        return f"{lhs} = {rhs}"


_get_expr = xt.elements._get_expr
_is_ref = xd.refs.is_ref

# build variables part
vars_str = ""
for vv in line.vars.keys():
    if vv == '__vary_default':
        continue
    vars_str += mad_assignment(vv, line.vars[vv]) + ";\n"
