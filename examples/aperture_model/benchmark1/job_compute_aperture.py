import matplotlib.pyplot as plt
from pyoptics import BeamEnvelope
import xobjects as xo
import xtrack as xt
from xtrack.aperture import Aperture

context = xo.ContextCpu(omp_num_threads="auto")

lhc_with_metadata = xt.load("./lhc_aperture.json")

b1 = lhc_with_metadata["b1"]
lhc_length = b1.get_length()

aperture_model = Aperture.from_line_with_madx_metadata(b1, num_profile_points=100, context=context)

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
    }
)

# Beam Envelope pyoptics vs Xtrack aperture model
ap = BeamEnvelope.from_apname("temp/ap_ir5b1.tfs")
ap.plot_aper_sx()
ap.plot_beam_sx()
plt.show()

# Cross-sections and beam pyoptics vs Xtrack aperture model
for name in ["MQXFA.A1R5", "MBXF.4R5"]:
    ap.plot_halo_name(name)
    xs_name = b1.get_table().rows[f'{name.lower()}.*'].name[0]

    sigmas, twiss, cross_sections, max_envelope = aperture_model.get_aperture_sigmas_at_element(
        element_name=xs_name,
        resolution=0.1,
        method='bisection',
    )

    cross_sections2, poses = aperture_model.cross_sections_at_element(element_name=xs_name, resolution=0.1)

    for pt in cross_sections2:
        plt.plot(pt[:, 0], pt[:, 1], c='gray', linestyle='--')

    for pt in max_envelope:
        plt.plot(pt[:, 0], pt[:, 1])

    plt.gca().set_aspect('equal')
    plt.title(f"{xs_name}")
    plt.legend()
    plt.show()

ap.ap.show("MQXFA.A1R5", "s n1 betx bety x y dx dy")
ap.ap.show("MBXF.4R5", "n1 x y betx bety x y dx dy")
