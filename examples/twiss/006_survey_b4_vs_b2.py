import numpy as np
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import xtrack as xt
import xpart as xp

mad_b2 = Madx()
mad_b2.call("../../test_data/hllhc15_noerrors_nobb/sequence.madx")
mad_b2.use(sequence="lhcb2")
twb2mad = mad_b2.twiss()
summb2mad = mad_b2.table.summ

mad_b4 = Madx()
mad_b4.call("../../test_data/hllhc15_noerrors_nobb/sequence_b4.madx")
mad_b4.use(sequence="lhcb2")
twb4mad = mad_b4.twiss()
summb4mad = mad_b4.table.summ

line_b4 = xt.Line.from_madx_sequence(
    mad_b4.sequence["lhcb2"],
    # deferred_expressions=True
)
line_b4.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7000e9)


line_b4.build_tracker()
twb4xt = line_b4.twiss()


# Survey with offset:
starting = {
    "theta0": -np.pi / 9,
    "psi0": np.pi / 7,
    "phi0": np.pi / 11,
    "X0": -300,
    "Y0": 150,
    "Z0": -100,
}
#starting = {}

survb4mad = mad_b4.survey(**starting).dframe()
survb2mad = mad_b2.survey(**starting).dframe()
survb4xt = line_b4.survey(**starting).to_pandas(index="name")
survb2xt = line_b4.survey(**starting, reverse=True).to_pandas(index="name")

# ================================
plt.close('all')
plt.figure(1, figsize=(6.4*1.4, 4.8*1.2))

sp_order = [1, 3, 5, 2, 4, 6]

for ii, coordi in enumerate("X Y Z theta phi psi".split()):
    plt.subplot(3, 2, sp_order[ii])

    plt.plot(survb2mad["s"], survb2mad[coordi.lower()], "r", lw=5, alpha=0.5)
    plt.plot(survb2mad.loc["ip2", "s"], survb2mad.loc["ip2", coordi.lower()],
            marker="o", color="red", ms=5, alpha=0.5)
    plt.plot(survb2xt["s"], survb2xt[coordi], "-", color="darkred", lw=2)
    plt.plot(survb2xt.loc["ip2", "s"], survb2xt.loc["ip2", coordi],
            marker="x", color='darkred')

    plt.plot(survb4mad["s"], survb4mad[coordi.lower()], "b", lw=5, alpha=0.5)
    plt.plot(survb4mad.loc["ip2", "s"], survb4mad.loc["ip2", coordi.lower()],
            marker="o", color="blue", ms=5, alpha=0.5)
    plt.plot(survb4xt["s"], survb4xt[coordi], "-", color="darkblue", lw=2)
    plt.plot(survb4xt.loc["ip2", "s"], survb4xt.loc["ip2", coordi],
            marker="x", color='darkblue')

    plt.ylabel(coordi)
plt.subplots_adjust(left=.105, right=.95, wspace=.23)
plt.show()