import matplotlib.pyplot as plt
import numpy as np

import xobjects as xo
import xtrack as xt

from xtrack.aperture import Aperture

context = xo.ContextCpu(omp_num_threads='auto')

lhc_with_metadata = xt.load('./lhc_aperture.json')

b1 = lhc_with_metadata['b1']
lhc_length = b1.get_length()

aperture_model = Aperture.from_line_with_madx_metadata(b1, num_profile_points=100, context=context)

emittance_norm=2.5e-6;
apbbeat=1.1;
DParcx=0.10; DParcy=0.10;
COmax=0.002; dPmax=0.0002; VMAXI=30; SPECIF=7;

aperture_model.halo_params.update(
{'emitx_norm': emittance_norm,
 'emity_norm': emittance_norm,
 'delta_rms': dPmax,
 'tol_co': COmax,
 'tol_disp': DParcx, #MADX has different setttings for x/y
 'tol_disp_ref_dx': 2.086,
 'tol_disp_ref_beta': 170.25,
 'tol_energy': 0.0, # TOCHECK
 'tol_beta_beating': apbbeat,  #MADX has different  setttings for x/y
})

from pyoptics import BeamEnvelope

ap=BeamEnvelope.from_apname("temp/ap_ir5b1.tfs")
ap.plot_aper_sx()
ap.plot_beam_sx()

ap.plot_halo_name("MQXFA.A1R5")
ap.plot_halo_name("MBXF.4R5")

ap.ap.show("MQXFA.A1R5","n1 betx bety x y dx dy")
ap.ap.show("MBXF.4R5","n1 x y betx bety x y dx dy")

