from cpymad.madx import Madx
import xtrack as xt
import xdeps as xd

mad = Madx()
# Element definitions
mad.input("""

a = 1.;
b := sin(3*a) + cos(2*a);

cav1: rfcavity, freq:=a*10, lag:=a*0.5, volt:=a*6;
cav2: rfcavity, freq:=10, lag:=0.5, volt:=6;
testseq: sequence, l=10;
c1: cav1, at=0.2, apertype=circle, aperture=0.01;
c2: cav2, at=0.5, apertype=circle, aperture=0.01;
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

# mad = Madx()
# folder = ('../../test_data/elena')
# mad.call(folder + '/elena.seq')
# mad.call(folder + '/highenergy.str')
# mad.call(folder + '/highenergy.beam')
# mad.use('elena')

# # Build xsuite line
# seq = mad.sequence.elena
# line = xt.Line.from_madx_sequence(seq)
# line.particle_ref = xt.Particles(gamma0=seq.beam.gamma,
#                                     mass0=seq.beam.mass * 1e9,
#                                     q0=seq.beam.charge)

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
    vv = _ge(var)
    if _is_ref(vv):
        out = expr_to_mad_str(vv)
        out = out.strip('._expr')
        return out
    else:
        return vv

def mad_assignment(lhs, rhs):
    if _is_ref(rhs):
        rhs = mad_str_or_value(rhs)
    if isinstance(rhs, str):
        return f"{lhs} := {rhs}"
    else:
        return f"{lhs} = {rhs}"


_ge = xt.elements._get_expr
_is_ref = xd.refs.is_ref

# build variables part
vars_str = ""
for vv in line.vars.keys():
    if vv == '__vary_default':
        continue
    vars_str += mad_assignment(vv, line.vars[vv]) + ";\n"


def cavity_to_madx_str(name, container):
    cav = container[name]
    tokens = []
    tokens.append('rfcavity')
    tokens.append(mad_assignment('freq', _ge(cav.frequency) * 1e-6))
    tokens.append(mad_assignment('volt', _ge(cav.voltage) * 1e-6))
    tokens.append(mad_assignment('lag', _ge(cav.lag) / 360.))

    return ', '.join(tokens)

def marker_to_madx_str(name, container):
    return 'marker'

def drift_to_madx_str(name, container):
    drift = container[name]
    tokens = []
    tokens.append('drift')
    import pdb; pdb.set_trace()
    tokens.append(mad_assignment('l', _ge(drift.length)))
    return ', '.join(tokens)

def multipole_to_madx_str(name, container):
    mult = container[name]

    # correctors are not handled correctly!!!!
    # https://github.com/MethodicalAcceleratorDesign/MAD-X/issues/911

    tokens = []
    tokens.append('multipole')
    knl_mad = []
    ksl_mad = []
    for kl, klmad in zip([mult.knl, mult.ksl], [knl_mad, ksl_mad]):
        for ii in range(len(kl._value)):
            item = mad_str_or_value(_ge(kl[ii]))
            if not isinstance(item, str):
                item = str(item)
            klmad.append(item)
    tokens.append('knl:={' + ','.join(knl_mad) + '}')
    tokens.append('ksl:={' + ','.join(ksl_mad) + '}')
    tokens.append(mad_assignment('lrad', _ge(mult.length)))

    return ', '.join(tokens)




xsuite_to_mad_conveters={
    xt.Cavity: cavity_to_madx_str,
    xt.Marker: marker_to_madx_str,
    xt.Drift: drift_to_madx_str,
    xt.Multipole: multipole_to_madx_str,
}

elements_str = ""
for nn in line.element_names:
    el = line[nn]
    el_str = xsuite_to_mad_conveters[type(el)](nn, line.element_refs)
    elements_str += f"{nn}: {el_str};\n"

print(elements_str)

line_str = 'myseq: line=(' + ', '.join(line.element_names) + ');'

mad_input = vars_str + '\n' + elements_str + '\n' + line_str

mad2 = Madx()
mad2.input(mad_input)
mad2.beam()
mad2.use('myseq')