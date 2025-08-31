import pymadng as ng
import numpy as np
from cpymad.madx import Madx

# MAD-NG
mng = ng.MAD()
X0 = [
    [0, 0, 0.001, 0, 0, 0],
]

phi = np.deg2rad(1)
mng['phi'] = phi
mng['X0'] = X0

ng_script = """
local sequence, quadrupole in MAD.element
local track, beam in MAD

quad = quadrupole 'quad' { l=10, k1=0.01, misalign={dphi=phi} }

local seq = sequence 'seq' { refer='entry',
    quad { at=0 }
}

tbl = track {X0=X0, sequence=seq, nturn=1, beam=beam { particle='proton' }, observe=0, save='atall'}
"""
mng.send(ng_script)
data = mng.tbl.to_df()


# MAD-X
mad = Madx()

madx_particles = '\n'.join([
    f'start, x={x}, px={px}, y={y}, py={py}, t={t}, pt={pt};'
    for x, px, y, py, t, pt in X0
])

mad.input(f"""
    beam, particle=electron, energy=1.0;
    phi = {phi};

    quad: quadrupole, l=10, k1=0.01, dphi=phi;

    seq: sequence, l=10, refer=entry;
        quad, at=0;
    endsequence;

    use, sequence=seq;

    track;
        {madx_particles}
        run, turns=1;
    endtrack;
""")

turn = mad.table.tracksumm.turn
mask = np.where(turn == 1)

for coord in ['x', 'px', 'y', 'py']:
    mng_val = list(data[data['name'] == '$end'][coord])
    mx_val = getattr(mad.table.tracksumm, coord)[mask]
    print(f"MAD-NG {coord}: {mng_val}          MAD-X {coord}: {mx_val}")
