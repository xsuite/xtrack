import matplotlib.pyplot as plt
from pyoptics import BeamEnvelope
import xobjects as xo
import xtrack as xt
from xtrack.aperture import Aperture
import numpy as np

context = xo.ContextCpu(omp_num_threads="auto")

lhc_with_metadata = xt.load("./lhc_aperture.json")

b1 = lhc_with_metadata["b1"]
lhc_length = b1.get_length()

aperture_model = Aperture.from_line_with_madx_metadata(b1, num_profile_points=100, include_offsets=True, context=context)

emittance_norm = 2.5e-6
apbbeat = 1.1
DParcx = 0.10
DParcy = 0.10
COmax = 0.002
dPmax = 0.0002
VMAXI = 30
SPECIF = 7

aperture_model.halo_params.update(
    {
        "emitx_norm": emittance_norm,
        "emity_norm": emittance_norm,
        "delta_rms": dPmax,
        "tol_co": COmax,
        "tol_disp": DParcx,  # MADX has different settings for x/y
        "tol_disp_ref_dx": 2.086,
        "tol_disp_ref_beta": 170.25,
        "tol_energy": 0.0,  # TO CHECK
        "tol_beta_beating": apbbeat,  # MADX has different  settings for x/y
        "halo_x": 1,
        "halo_y": 1,
        "halo_r": 1,
        "halo_primary": 1,
    }
)

# Beam Envelope pyoptics vs Xtrack aperture model
ap = BeamEnvelope.from_apname("temp/ap_ir5b1.tfs")

tt = aperture_model.get_bounds_table()

# Cross-sections and beam pyoptics vs Xtrack aperture model
# ["MQXFA.A1R5", "MBXF.4R5", "TAXN.4L5"]
#name = "MQXFA.A1R5"
name = "MBXF.4R5"

pyop_id = ap.get_n_name(name)[0]
n_sigma = ap.get_n1_name(name)[0][0]
s = ap.ap.s[pyop_id]

ap.plot_halo(pyop_id, halor=n_sigma, halox=n_sigma, haloy=n_sigma)

envel, tw = aperture_model.get_envelope_at_s(s_positions=[s], sigmas=n_sigma, include_aper_tols=True, envelopes_num_points=101)

cross_sections, poses = aperture_model.cross_sections_at_s(s_positions=[s])


ap_centre = (np.min(cross_sections, axis=1) + np.max(cross_sections, axis=1)) / 2

for pt, ct in zip(cross_sections, ap_centre):
    plt.plot(pt[:, 0] - ct[0], pt[:, 1] - ct[1], c='gray', linestyle='--')

for pt, ct in zip(envel, ap_centre):
    plt.plot(pt[:, 0] - ct[0], pt[:, 1] - ct[1], c='b', linestyle='-', marker='.')

plt.gca().set_aspect('equal')
plt.title(f"s = {s}")
plt.legend()
plt.show()

ap.ap.show(name, "s n1 betx bety x y dx dy")
print(tt.rows[f'{name.lower()}.*'])
