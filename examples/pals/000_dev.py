import xtrack as xt
import yaml
from pathlib import Path

b = xt.Bend(length=5, angle=0.1, k0=0.001)

out_pals = {}
out_pals['length'] = float(b.length)

out_pals['BendP'] = {
    'angle_ref': float(b.angle),
}

out_pals['MagneticMultipoleP'] = {
    'Kn1L': float(b.k0)
}

with open(Path(__file__).parent / 'pals.yaml', 'w') as fid:
    yaml.safe_dump(out_pals, fid, sort_keys=False)
