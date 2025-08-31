from cpymad.madx import Madx
import xtrack as xt


madx = Madx()

mad_data = """

crf: rfcavity, volt=3, freq=400, lag=0.1;
ccrab: rfmultipole, volt=3, freq=400, lag=0.;

ss_cav: sequence, l=3;
  cc1: crf, at=1;
endsequence;

ss_crab: sequence, l=3;
  cc2: ccrab, at=1;
endsequence;

"""

mad_computation_cav = """
beam, bv=1, sequence=ss_cav;
use, sequence=ss_cav;
twiss, betx=1, bety=1, t=0.1;
"""
madx.input(mad_data)
madx.input(mad_computation_cav)
tw_mad_cav_bv1 = xt.Table(madx.table.twiss, _copy_cols=True)

mad_computation_cav = """
beam, bv=-1, sequence=ss_cav;
use, sequence=ss_cav;
twiss, betx=1, bety=1, t=0.1;
"""
madx.input(mad_computation_cav)
tw_mad_cav_bvm1 = xt.Table(madx.table.twiss, _copy_cols=True)

# mad_computation_crab = """
# beam, bv=-1, sequence=ss_crab;
# use, sequence=ss_crab;
# twiss, betx=1, bety=1, t=-0.1;
# """

# madx.input(mad_computation_crab)
# tw_mad_crab = xt.Table(madx.table.twiss, _copy_cols=True)

print(f"RF Cavity bv1 pt={tw_mad_cav_bv1['pt'][-1]}, t={tw_mad_cav_bv1['t'][-1]}")
print(f"RF Cavity bv-1 pt={tw_mad_cav_bvm1['pt'][-1]}, t={tw_mad_cav_bvm1['t'][-1]}")