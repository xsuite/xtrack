import xtrack as xt
import numpy as np
from cpymad.madx import Madx
import pymadng as ng

dx = 0
dy = 0
ds = 0
theta = 0.0
phi = 0.5
psi = 0.0

# Xtrack
p0 = xt.Particles(x=[0.01, -0.01], px=[0.02, -0.02], y=[0.03, -0.03], py=[0.04, -0.04])
line = xt.Line(elements=[
    xt.XYShift(dx=dx, dy=dy),
    xt.Solenoid(length=ds),
    xt.YRotation(angle=np.rad2deg(theta)),
    xt.XRotation(angle=np.rad2deg(phi)),
    xt.SRotation(angle=np.rad2deg(psi)),
    xt.Marker(),
])
p = p0.copy()
line.track(p)

# MAD-X
mad = Madx()

madx_particles = '\n'.join([
    f'start, x={p0.x[i]}, px={p0.px[i]}, y={p0.y[i]}, py={p0.py[i]};'
    for i in range(len(p0.x))
])

mad.input(f"""
    beam, particle=electron, energy=1.0;
    dx = {dx};
    dy = {dy};
    ds = {ds};
    theta = {theta};
    phi = {phi};
    psi = {psi};

    shift: translation, dx=dx, dy=dy, ds=ds;
    yrot: yrotation, angle=theta;
    xrot: xrotation, angle=phi;
    srot: srotation, angle=psi;
    end: marker;

    seq: sequence, l=0.0000000001;
        ! shift, at=0;
        yrot, at=0;
        xrot, at=0;
        srot, at=0;
        end, at=0;
    endsequence;

    use, sequence=seq;

    track;
        {madx_particles}
        run, turns=1;
    endtrack;
""")

mx_number = mad.table.tracksumm.number
mx_turn = mad.table.tracksumm.turn
mx_table = mad.table.tracksumm
mx_mask = np.where(mx_turn == 1)


# MAD-NG
mng = ng.MAD()
mng['dx'] = dx
mng['dy'] = dy
mng['ds'] = ds
mng['theta'] = theta
mng['phi'] = phi
mng['psi'] = psi

X0 = [
    [p0.x[0], p0.px[0], p0.y[0], p0.py[0]],
    [p0.x[1], p0.px[1], p0.y[1], p0.py[1]],
]

mng['X0'] = X0

ng_script = """
local sequence, marker, srotation, xrotation, yrotation, translate in MAD.element
local track, beam in MAD

shift = translate 'shift' { dx=dx, dy=dy, ds=ds }
yrot = yrotation 'yrot' { angle=theta }
xrot = xrotation 'xrot' { angle=phi }
srot = srotation 'srot' { angle=psi }

local seq = sequence 'seq' { refer='centre',
    shift { at=0 },
    yrot { at=0 },
    xrot { at=0 },
    srot { at=0 },
}

tbl = track {X0=X0, sequence=seq, nturn=1, beam=beam{}, observe=0}
"""
mng.send(ng_script)
data = mng.tbl.to_df()
print(data)

for coord in ['x', 'px', 'y', 'py']:
    ng_val = list(data[data['name'] == '$end'][coord])
    mx_val = getattr(mx_table, coord)[mx_mask]
    print(f"Xtrack {coord}: {getattr(p, coord)}")
    print(f"=> MAD-NG {coord}: {ng_val}, error: {np.max(np.abs(getattr(p, coord) - ng_val))}")
    print(f"=> MAD-X {coord}: {mx_val}, error: {np.max(np.abs(getattr(p, coord) - mx_val))}")
