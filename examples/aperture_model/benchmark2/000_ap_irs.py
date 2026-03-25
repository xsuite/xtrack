import xtrack as xt
from xtrack.aperture import Aperture
from xobjects import ContextCpu
import matplotlib.pyplot as plt
from lhcoptics import LHCOptics
import numpy as np
from xdeps import Table

base = "./acc-models-lhc/" # "https://cern.ch/acc-models/lhc/hl19/"
lhc = xt.load(f"{base}/xsuite/lhc_aperture.json")
lhc.vars.load(f"{base}/strengths/cycle_round_v0/opt_6000.madx")
lhc.set_particle_ref(p0c=450e9)

# load in metadata
lhc.b1.metadata["aperture_offsets"] = {}
lhc.b2.metadata["aperture_offsets"] = {}
for ipn in range(1, 9):
    for beam in "14":
        tfs = Table.from_tfs(f"./temp/offset.ip{ipn}.b{beam}.tfs")
        line = lhc.b1 if beam == "1" else lhc.b2
        line.metadata["aperture_offsets"][f"ip{ipn}"] = tfs._data.copy()

opt = LHCOptics.from_xsuite(lhc)

mad = opt.make_madx_model()
apm = mad.get_ap_irs()
ir4 = apm['ir4b1']
# mad.get_ap_arc('12', '1')

# all sort of params in ap, including ap.s which we should use for benchmark slicing
context = ContextCpu(omp_num_threads='auto')
b1 = lhc.b1
apx = Aperture.from_line_with_madx_metadata(b1, num_profile_points=100, include_offsets=True, context=context)

apx.halo_params.update(
    {
        "emitx_norm": ir4.exn,
        "emity_norm": ir4.eyn,
        "delta_rms": ir4.dp_bucket_size,
        "tol_co": ir4.co_radius,
        "tol_disp": ir4.paras_dx,  # MADX has different settings for x/y
        "tol_disp_ref_dx": ir4.dqf,
        "tol_disp_ref_beta": ir4.betaqfx,
        "tol_energy": 0.0,  # TO CHECK
        "tol_beta_beating": ir4.beta_beating,  # MADX has different  settings for x/y
    }
)

s_ip4_m, = ir4.rows['ip4.*'].s
s_ip4_x, = b1.get_table().rows['ip4.*'].s
s_positions = ir4.s - s_ip4_m + s_ip4_x

n1_table, twiss = apx.get_aperture_sigmas_at_s(
    s_positions=s_positions,
    method='rays',
)
sigmas = n1_table.n1

plt.figure()
plt.plot(s_positions, np.where(ir4.n1 > 9e5, np.inf, ir4.n1), label='n1 (MAD-X)')
plt.plot(s_positions, sigmas, label='n1 (Xtrack)')
plt.xlabel('s [m]')
plt.ylabel(r'max beam size [$\sigma$]')
plt.legend()
plt.show()
