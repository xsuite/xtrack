import cernlayoutdb as layout
import numpy as np
import xtrack as xt
from matplotlib import pyplot as plt

lhc = layout.Machine.from_pickle("LHC.pickle")
ldb_curv = lhc.get_ref_curve()

lattice = xt.load('https://acc-models.web.cern.ch/acc-models/lhc/hl19/xsuite/lhc.json')
b1 = lattice.b1
b2 = lattice.b2

sv1 = b1.survey()
sv2 = b2.survey(reverse=False).reverse()

ldb_mid_points = np.array([[mpt.x, mpt.z] for pt in ldb_curv.points if (mpt := pt.to_madpoint())])
ldb_mid_X = ldb_mid_points[:, 0]
ldb_mid_Z = ldb_mid_points[:, 1]

plt.title('HL-LHC 1.9 Survey')
plt.plot(ldb_mid_Z, ldb_mid_X, 'og-', label='LDB reference curve')
plt.plot(sv1.Z, sv1.X, 'o-b', label='Xsuite beam 1')
plt.plot(sv2.Z, sv2.X, 'o-r', label='Xsuite beam 2')
plt.xlabel('Z [m] (MAD convention)')
plt.ylabel('X [m] (MAD convention)')
plt.legend()
plt.show()
