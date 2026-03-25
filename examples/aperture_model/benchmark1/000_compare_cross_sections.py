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
with open('lhc_aperture_model.json', 'w') as f:
    xt.json.dump(aperture_model.model.to_dict(), f)

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

tt = aperture_model.get_bounds_table()

# Cross-sections and beam pyoptics vs Xtrack aperture model
for name in ["MB.A9L5", "MQXFA.A1R5", "MBXF.4R5", "TAXN.4L5"]:
    ap.plot_halo_name(name)

    xs_name = b1.get_table().rows[f'{name.lower()}.*'].name[0]

    n1_table, twiss = aperture_model.get_aperture_sigmas_at_element(
        element_name=xs_name,
        resolution=0.1,
        method='bisection',
        output_max_envelopes=True,
    )
    sigmas = n1_table.n1
    max_envelope = n1_table.envelope

    n1_rays, _ = aperture_model.get_aperture_sigmas_at_element(
        element_name=xs_name,
        resolution=0.1,
        method='rays',
    )
    sigmas_rays = n1_rays.n1
    n1_exact, _ = aperture_model.get_aperture_sigmas_at_element(
        element_name=xs_name,
        resolution=0.1,
        method='exact',
        output_max_envelopes=True,
    )
    sigmas_exact = n1_exact.n1
    max_envelope_exact = n1_exact.envelope

    cross_sections_table = aperture_model.cross_sections_at_element(element_name=xs_name, resolution=0.1)
    cross_sections = cross_sections_table.cross_section


    ap_centre = (np.min(cross_sections, axis=1) + np.max(cross_sections, axis=1)) / 2

    for pt, ct in zip(cross_sections, ap_centre):
        plt.plot(pt[:, 0] - ct[0], pt[:, 1] - ct[1], c='gray', linestyle='--')

    seen = False
    for pt, ct in zip(max_envelope, ap_centre):
        plt.plot(pt[:, 0] - ct[0], pt[:, 1] - ct[1], c='cyan', linestyle=':', label='halo (Xtrack)' if not seen else '')
        seen = True

    plt.gca().set_aspect('equal')
    plt.suptitle(f"{xs_name}")
    plt.title(
        " / ".join(
            [
                f"min(n1_rays) = {min(sigmas_rays):.5f}",
                f"min(n1_bisection) = {min(sigmas):.5f}",
                f"min(n1_exact) = {min(sigmas_exact):.5f}",
                f"min(n1_pyoptics) = {min(n1_pyoptics):.5f}",
            ]
        )
    )
    plt.legend()
    plt.show()

    ap.ap.show(name, "s n1 betx bety x y dx dy")
    print(tt.rows[f'{name.lower()}.*'])
