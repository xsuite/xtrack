import xtrack as xt
import xdeps
import numpy as np
import re

from xdeps.madxutils import to_madx

stmt = re.compile(r"([a-z0-9._]+)(,|:=|:|=)(.*);")


def parse_command(stmt):
    groups = []
    rest = stmt
    while len(rest) > 0:
        if res := re.match(r"([a-z0-9._]+)[,;]", rest):
            groups.append(res.groups())
        elif res := re.match(r"([a-z0-9._]+)(=|:=)(\{[^\}]+\})[,;]", rest):
            groups.append(res.groups())
        elif res := re.match(r"([a-z0-9._]+)(=|:=)([^,]+)[,;]", rest):
            groups.append(res.groups())
        else:
            print(rest)
            raise ValueError("Input Error {stmt}")
        rest = rest[len(res.group()) :]
    return groups


# examples=["fd,b={4,6},a=5;"]
# for ex in examples:
#    print(parse_command(ex))


def parse_madx_expr(av):
    av_expr = env._xdeps_eval.eval(av)
    if not isinstance(av_expr, float):
        return av
    else:
        return av_expr


def invert_at(groups):
    out = []
    for an, eq, av in groups:
        if an == "at":
            av = f"lhclength-({av})"
        out.append((an, eq, av))
    return out


def groups_to_attrs(groups):
    attrs = {}
    extra = {}
    for an, eq, av in groups:
        if av.startswith("{"):
            av = [parse_madx_expr(aa) for aa in av[1:-1].split(",")]
        else:
            av = parse_madx_expr(av)
        if an in [
            "slot_id",
            "mech_sep",
            "assembly_id",
            "kmin",
            "kmax",
            "calib",
            "polarity",
        ]:
            extra[an] = (av, eq)
        else:
            if an == "l" or an == "lrad":
                an = "length"
            if an == "tilt":
                an = "rot_s_rad"
            attrs[an] = (av, eq)
    attrs = ",".join([f"{an}={av!r}" for an, (av, eq) in attrs.items()])
    extra = ",".join([f"{an}={av!r}" for an, (av, eq) in extra.items()])
    out = []
    if attrs:
        out.append(f"{attrs}")
    if extra:
        out.append(f"extra=dict({extra})")
    return "," + ",".join(out)


def default_for_zeros(expr, vv, notinitialized):
    if hasattr(expr, "_get_dependencies"):
        for ll in expr._get_dependencies():
            if hasattr(ll, "_key"):
                if ll._key not in vv:
                    vv[ll._key] = 0
                    notinitialized.add(ll._key)


env = xt.Environment()
env.vars["twopi"] = np.pi * 2

lines = [
    "import xtrack as xt",
    "import numpy as np",
    "env=xt.Environment()",
    "env._xdeps_vref._owner.default_factory = lambda : 0",
    "env.vars['twopi']=np.pi*2",
    "env.new('vkicker','Multipole')",
    "env.new('hkicker','Multipole')",
    "env.new('tkicker','Multipole')",
    "env.new('collimator','Drift')",
    "env.new('instrument','Drift')",
    "env.new('monitor','Drift')",
    "env.new('placeholder','Drift')",
    "env.new('sbend','Bend')",
    "env.new('rbend','Bend')",
    "env.new('quadrupole','Quadrupole')",
    "env.new('sextupole','Sextupole')",
    "env.new('octupole','Octupole')",
    "env.new('marker','Drift')",
    "env.new('rfcavity','Cavity')",
    "env.new('multipole','Multipole', knl=[0, 0, 0, 0, 0, 0])",
    "env.new('solenoid','Solenoid')",
]

notparsed = []
notinitialized = set()
inside_sequence = False
lhcb2 = False
for line in open("lhc.seq"):

    if '/*                       ACSCA CAVITIES ' in line:
        break

    line = line.replace(", L := l.OMK;", ";")
    line = line.replace(" L := l.ACSCA,", "")
    line = line.replace(", HARMON := HRF400", "")
    if "INSTRUMENT" in line or "PLACEHOLDER" in line:
        if "lrad" in line.lower():
            newline = line.lower()
            newline = newline.split("lrad")[0]
            newline = newline.strip()
            newline = newline.strip(",")
            newline = newline + ";"
            line = newline

    line = line.strip()
    ls = line.replace(" ", "").lower()
    if res := stmt.match(ls):
        lhs, eq, rhs = res.groups()
        if eq == ":=":  # variable
            expr = env._xdeps_eval.eval(rhs)
            default_for_zeros(expr, env.vars, notinitialized)
            env.vars[lhs] = expr
            if not isinstance(expr, float):
                expr = rhs
            lines.append(f"env[{lhs!r}]={expr!r}")
        elif eq == ":":  # element definition
            groups = parse_command(rhs + ";")
            parent = groups.pop(0)[0]
            if parent == "sequence":
                lines.append(f"{lhs}=env.new_builder(name={lhs!r})")
                inside_sequence = True
                sequence_name=lhs
                if lhs == "lhcb2":
                    lhcb2 = True
            else:
                groups = [
                    (an.replace("from", "from_"), eq, av) for an, eq, av in groups
                ]
                if lhcb2:
                    groups = invert_at(groups)
                attrs = groups_to_attrs(groups)
                if inside_sequence:
                    lines.append(f"{sequence_name}.new({lhs!r:20},{parent!r}{attrs})")
                else:
                    lines.append(f"env.new({lhs!r:20},{parent!r}{attrs})")
        elif eq == ",":  # element definition
            groups = parse_command(rhs + ";")
            attrs = groups_to_attrs(groups)
            lines.append(f"env.set({lhs!r:20}{attrs})")
        else:
            lines.append("#" + line)
    else:
        if line.startswith("ENDSEQUENCE"):
            if lhcb2:
               lines.append(f"#{sequence_name}.reflect().build() ")
            else:
               lines.append(f"{sequence_name}.build()")
            inside_sequence = False
        elif len(line) > 0:
            if not line.startswith("if") or not line.startswith("return"):
                lines.append("#" + line)
        else:
            lines.append("")

import re

for ii in range(len(lines)):
    line = lines[ii]
    if ",kick" in line:
        if "h." in line:
            line = re.sub(r"kick='([-a-z0-9.]+)'", r"knl=['-(\1)']", line)
        else:
            line = re.sub(r"kick=('[-a-z0-9.]+')", r"ksl=[\1]", line)
    if ",hkick" in line:
        line = re.sub(r"hkick='([-a-z0-9.]+)'", r"knl=['-(\1)']", line)
    if ",vkick" in line:
        line = re.sub(r"vkick=('[-a-z0-9./]+')", r"ksl=[\1]", line)
    lines[ii] = line

for ll in lines:
    print(ll)
