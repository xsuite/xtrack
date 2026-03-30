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
    xt.json.dump(aperture_model._model.to_dict(), f)

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
for name in ["MB.B10L5", "MB.A9L5", "MQXFA.A1R5", "MBXF.4R5", "TAXN.4L5"]:
    n1_pyoptics = np.min([row[0] for row in ap.get_n1_name(name)])

    xs_name = b1.get_table().rows[f'{name.lower()}.*'].name[0]

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    aperture_model.plot_n1_at_element(xs_name, method='rays', middle='aperture', ax=axs[0, 0])
    aperture_model.plot_n1_at_element(xs_name, method='bisection', middle='aperture', ax=axs[0, 1])
    aperture_model.plot_n1_at_element(xs_name, method='exact', middle='aperture', ax=axs[1, 0])
    plt.sca(axs[1, 1])
    ap.plot_halo_name(name)

    for ax in axs.flat:
        ax.set(xlabel='m', ylabel='m')

    for ax in axs.flat:
        ax.label_outer()
    plt.show()


    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    aperture_model.plot_at_element(xs_name, middle='aperture', method='rays', ax=axs[0, 0])
    aperture_model.plot_at_element(xs_name, middle='aperture', method='bisection', ax=axs[0, 1])
    aperture_model.plot_at_element(xs_name, middle='aperture', method='exact', ax=axs[1, 0])
    plt.sca(axs[1, 1])
    ap.plot_halo_name(name, n1=n1_pyoptics)

    for ax in axs.flat:
        ax.set(xlabel='m', ylabel='m')

    for ax in axs.flat:
        ax.label_outer()
    plt.show()
