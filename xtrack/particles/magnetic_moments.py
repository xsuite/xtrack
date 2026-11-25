# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Ground-state magnetic moments and charge-normalized BMT anomalies.

Generated offline by test_data/particle_data/generate_magnetic_moments.py; source
excerpts, calculations and conventions are in test_data/particle_data/README.md.
Regenerate from the repository root:
    python test_data/particle_data/generate_magnetic_moments.py

Sources:
  CODATA 2022: https://physics.nist.gov/cuu/Constants/Table/allascii.txt
  NUBASE2020: https://doi.org/10.1088/1674-1137/abddae
  IAEA recommended moments: https://doi.org/10.61092/iaea.qs1q-27sa

For fully stripped nuclei, a = mu*m/(2*I*Z*m_p)-1 with mu in nuclear magnetons.
None means unavailable or inapplicable, not zero. Unsigned moments do not
provide anomalies. Input qualifiers and uncertainties are retained in the
source excerpts. Derived decimal places are computational guard digits,
not measurement accuracy; correlated uncertainties are not propagated here.
"""
from typing import NamedTuple


# Ground-state spin data, generated offline. Sources and regeneration:
# test_data/particle_data/README.md. None means unavailable or inapplicable,
# not a zero magnetic moment. Anomalies use charge-normalized g factors.
class SpinProperties(NamedTuple):
    """Spin I [hbar], signed moment [mu_N], and charge-normalized BMT anomaly.

    Known spin zero and the known neutron moment are retained as data, but
    have no charge-normalized BMT anomaly. Photon helicity is not represented
    by this massive-particle spin record.
    """
    spin: float | None
    magnetic_moment_mu_N: float | None
    anomalous_magnetic_moment: float | None


# Same species order and element groups as masses.py.
# Fields: spin I [hbar], magnetic moment [mu_N], BMT anomaly (g-2)/2.

# Direct CODATA values and light nuclei
ELECTRON_SPIN_PROPERTIES     = SpinProperties(0.5, -1838.2819718826383, 0.00115965218046)  # CODATA 2022; sigma(a)=1.8e-13
MUON_SPIN_PROPERTIES         = SpinProperties(0.5, -8.890597037489346, 0.00116592062)  # CODATA 2022; sigma(a)=4.1e-10
PROTON_SPIN_PROPERTIES       = SpinProperties(0.5, 2.79284734463, 1.79284734463)  # CODATA 2022; sigma(mu/mu_N)=8.2e-10
DEUTERON_SPIN_PROPERTIES     = SpinProperties(1.0, 0.8574382335, -0.14298726968020162)  # CODATA 2022; sigma(mu/mu_N)=2.2e-9
TRITON_SPIN_PROPERTIES       = SpinProperties(0.5, 2.978962465, 7.918170675201702)  # CODATA 2022; sigma(mu/mu_N)=5.9e-9
He3_SPIN_PROPERTIES          = SpinProperties(0.5, -2.1276253498, -4.184153749907722)  # CODATA 2022; sigma(mu/mu_N)=1.7e-9
He4_SPIN_PROPERTIES          = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
NEUTRON_SPIN_PROPERTIES      = SpinProperties(0.5, -1.91304276, None)  # CODATA 2022; sigma(mu/mu_N)=4.5e-7

# Photon and mesons
PHOTON_SPIN_PROPERTIES       = SpinProperties(None, None, None)  # Photon helicity is outside the massive-particle BMT model
PION_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # PDG
KAON_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # PDG
PION0_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # PDG
KAON0_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # PDG

# C
C10_SPIN_PROPERTIES          = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
C11_SPIN_PROPERTIES          = SpinProperties(1.5, -0.964, -1.5852883413559977)  # IAEA; I=3/2-; mu/mu_N=-0.964(1)
C12_SPIN_PROPERTIES          = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
C13_SPIN_PROPERTIES          = SpinProperties(0.5, 0.702369, 0.5108136608855124)  # IAEA; I=1/2-; mu/mu_N=+0.702369(4)
C14_SPIN_PROPERTIES          = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
C15_SPIN_PROPERTIES          = SpinProperties(0.5, None, None)  # IAEA; I=1/2+; mu/mu_N=1.720(9); sign undetermined

# O
O14_SPIN_PROPERTIES          = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
O15_SPIN_PROPERTIES          = SpinProperties(0.5, None, None)  # IAEA; I=1/2-; mu/mu_N=0.71908(12); sign undetermined
O16_SPIN_PROPERTIES          = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
O17_SPIN_PROPERTIES          = SpinProperties(2.5, -1.893543, -1.7986953455197632)  # IAEA; I=5/2+; mu/mu_N=-1.893543(10)
O18_SPIN_PROPERTIES          = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
O19_SPIN_PROPERTIES          = SpinProperties(2.5, None, None)  # IAEA; I=5/2+; mu/mu_N=1.53238(7); sign undetermined
O20_SPIN_PROPERTIES          = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+

# Ne
Ne19_SPIN_PROPERTIES         = SpinProperties(0.5, -1.88515, -4.555236574173405)  # IAEA; I=1/2+; mu/mu_N=-1.88515(7)
Ne20_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Ne21_SPIN_PROPERTIES         = SpinProperties(1.5, -0.6617, -1.4595891778034686)  # IAEA; I=3/2+; mu/mu_N=-0.66170(3)
Ne22_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Ne23_SPIN_PROPERTIES         = SpinProperties(2.5, -1.0794, -1.4927010825424114)  # IAEA; I=5/2+; mu/mu_N=-1.0794(10)
Ne24_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+

# Ar
Ar35_SPIN_PROPERTIES         = SpinProperties(1.5, 0.632, -0.5937315321855616)  # IAEA; I=3/2+; mu/mu_N=+0.6320(2)
Ar36_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Ar37_SPIN_PROPERTIES         = SpinProperties(1.5, 1.146, -0.2213578515000751)  # IAEA; I=3/2+; mu/mu_N=+1.146(5)
Ar38_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Ar39_SPIN_PROPERTIES         = SpinProperties(3.5, -1.59, -1.4880170708355511)  # IAEA; I=7/2-; mu/mu_N=-1.590(15)
Ar40_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Ar41_SPIN_PROPERTIES         = SpinProperties(3.5, -1.31, -1.4227223328914171)  # IAEA; I=7/2-; mu/mu_N=-1.310(8)
Ar42_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Ar43_SPIN_PROPERTIES         = SpinProperties(2.5, -1.022, -1.4842629895786827)  # IAEA; I=5/2(-); mu/mu_N=-1.022(6)
Ar44_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Ar45_SPIN_PROPERTIES         = SpinProperties(None, None, None)  # NUBASE2020; I=(5/2-,7/2-)
Ar46_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Ar47_SPIN_PROPERTIES         = SpinProperties(None, None, None)  # NUBASE2020; I=(3/2)-

# Fe
Fe52_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Fe53_SPIN_PROPERTIES         = SpinProperties(3.5, -0.65, -1.1876739670672287)  # IAEA; I=7/2-; mu/mu_N=-0.65(1)
Fe54_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Fe55_SPIN_PROPERTIES         = SpinProperties(1.5, None, None)  # NUBASE2020; I=3/2-
Fe56_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Fe57_SPIN_PROPERTIES         = SpinProperties(0.5, 0.0902, -0.8039538625289329)  # IAEA; I=1/2-; mu/mu_N=+0.09020(4)
Fe58_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Fe59_SPIN_PROPERTIES         = SpinProperties(1.5, -0.3358, -1.2518285199506982)  # IAEA; I=3/2-; mu/mu_N=-0.3358(4)
Fe60_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Fe61_SPIN_PROPERTIES         = SpinProperties(None, None, None)  # NUBASE2020; I=(3/2-)
Fe62_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Fe63_SPIN_PROPERTIES         = SpinProperties(None, None, None)  # NUBASE2020; I=(5/2-)
Fe64_SPIN_PROPERTIES         = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+

# Xe
Xe112_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Xe113_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=5/2+#
Xe114_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Xe115_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=(5/2+)
Xe116_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Xe117_SPIN_PROPERTIES        = SpinProperties(2.5, -0.5938, -1.2552164362437186)  # IAEA; I=5/2+; mu/mu_N=-0.5938(15)
Xe118_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Xe119_SPIN_PROPERTIES        = SpinProperties(2.5, -0.6542, -1.2859754948582387)  # IAEA; I=5/2+; mu/mu_N=-0.6542(15)
Xe120_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Xe121_SPIN_PROPERTIES        = SpinProperties(2.5, -0.701, -1.3115784209037935)  # IAEA; I=5/2+; mu/mu_N=-0.701(3)
Xe122_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Xe123_SPIN_PROPERTIES        = SpinProperties(0.5, -0.15, -1.3388650271930391)  # IAEA; I=1/2+; mu/mu_N=-0.150(3)
Xe124_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Xe125_SPIN_PROPERTIES        = SpinProperties(0.5, -0.269, -1.6175785814497656)  # IAEA; I=1/2+; mu/mu_N=-0.269(3)
Xe126_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Xe127_SPIN_PROPERTIES        = SpinProperties(0.5, -0.5033, -2.1739867628799514)  # IAEA; I=1/2+; mu/mu_N=-0.5033(11)
Xe128_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Xe129_SPIN_PROPERTIES        = SpinProperties(0.5, -0.777961, -2.843254572976861)  # IAEA; I=1/2+; mu/mu_N=-0.777961(16)
Xe130_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Xe131_SPIN_PROPERTIES        = SpinProperties(1.5, 0.691845, -0.44511363974497276)  # IAEA; I=3/2+; mu/mu_N=+0.691845(7)
Xe132_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Xe133_SPIN_PROPERTIES        = SpinProperties(1.5, 0.81335, -0.3376890053415945)  # IAEA; I=3/2+; mu/mu_N=+0.81335(7)
Xe134_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Xe135_SPIN_PROPERTIES        = SpinProperties(1.5, 0.9032, -0.25344676404163574)  # IAEA; I=3/2+; mu/mu_N=+0.9032(7)
Xe136_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Xe137_SPIN_PROPERTIES        = SpinProperties(3.5, -0.9704, -1.3488647776611902)  # IAEA; I=7/2-; mu/mu_N=-0.9704(10)
Xe138_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Xe139_SPIN_PROPERTIES        = SpinProperties(1.5, -0.304, -1.2587491549644862)  # IAEA; I=3/2-; mu/mu_N=-0.3040(16)
Xe140_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Xe141_SPIN_PROPERTIES        = SpinProperties(2.5, 0.01, -0.9948192758420455)  # IAEA; I=5/2(-); mu/mu_N=+0.010(3)
Xe142_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+

# Au
Au176_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=(3-,4-)
Au177_SPIN_PROPERTIES        = SpinProperties(0.5, None, None)  # IAEA; I=1/2+; mu/mu_N=1.15(5); sign undetermined
Au178_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # IAEA; I=2,3; mu/mu_N=(I = 2) -0.88(7); no usable signed moment; uncertain I=(2+,3-)* (NUBASE2020)
Au179_SPIN_PROPERTIES        = SpinProperties(0.5, None, None)  # IAEA; I=1/2+; mu/mu_N=1.01(5); sign undetermined
Au180_SPIN_PROPERTIES        = SpinProperties(None, -0.83, None)  # IAEA; I=(1+); mu/mu_N=-0.83(9); uncertain I=(1+)* (NUBASE2020)
Au181_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=(5/2-)
Au182_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # IAEA; I=(2+); mu/mu_N=1.66(9); sign undetermined; uncertain I=(2+)* (NUBASE2020)
Au183_SPIN_PROPERTIES        = SpinProperties(2.5, 1.97, -0.09428190420186615)  # IAEA; I=(5/2)-; mu/mu_N=+1.97(10); I=5/2-* (NUBASE2020)
Au184_SPIN_PROPERTIES        = SpinProperties(5.0, 2.07, -0.5215522137564311)  # IAEA; I=5+; mu/mu_N=+2.07(10)
Au185_SPIN_PROPERTIES        = SpinProperties(2.5, 2.2, 0.022510711223903825)  # IAEA; I=5/2-; mu/mu_N=+2.20(6)
Au186_SPIN_PROPERTIES        = SpinProperties(3.0, -1.2, -1.467291339855468)  # IAEA; I=3-; mu/mu_N=-1.20(6)
Au187_SPIN_PROPERTIES        = SpinProperties(0.5, 0.56, 0.31544322513577716)  # IAEA; I=1/2+; mu/mu_N=+0.56(4)
Au188_SPIN_PROPERTIES        = SpinProperties(1.0, -0.07, -1.0826553495284104)  # IAEA; I=1(-); mu/mu_N=-0.07(3)
Au189_SPIN_PROPERTIES        = SpinProperties(0.5, 0.5, 0.1870659341734537)  # IAEA; I=1/2+; mu/mu_N=+0.50(3)
Au190_SPIN_PROPERTIES        = SpinProperties(1.0, -0.065, -1.0775680344174225)  # IAEA; I=1-; mu/mu_N=-0.065(7)
Au191_SPIN_PROPERTIES        = SpinProperties(1.5, 0.1363, -0.890993505854949)  # IAEA; I=3/2+; mu/mu_N=+0.1363(19)
Au192_SPIN_PROPERTIES        = SpinProperties(1.0, -0.0107, -1.0129033608587428)  # IAEA; I=1-; mu/mu_N=-0.0107(15)
Au193_SPIN_PROPERTIES        = SpinProperties(1.5, 0.1398, -0.8870228946577805)  # IAEA; I=3/2+; mu/mu_N=+0.1398(15)
Au194_SPIN_PROPERTIES        = SpinProperties(1.0, 0.0754, -0.908125692591862)  # IAEA; I=1-; mu/mu_N=+0.0754(25)
Au195_SPIN_PROPERTIES        = SpinProperties(1.5, None, None)  # IAEA; I=3/2+; mu/mu_N=0.1481(6); sign undetermined
Au196_SPIN_PROPERTIES        = SpinProperties(2.0, None, None)  # IAEA; I=2-; mu/mu_N=0.5883(5); sign undetermined
Au197_SPIN_PROPERTIES        = SpinProperties(1.5, 0.1452, -0.8802245633395173)  # IAEA; I=3/2+; mu/mu_N=+0.1452(2)
Au198_SPIN_PROPERTIES        = SpinProperties(2.0, 0.5911, -0.6324411750251565)  # IAEA; I=2-; mu/mu_N=+0.5911(8)
Au199_SPIN_PROPERTIES        = SpinProperties(1.5, 0.2705, -0.7745959314900602)  # IAEA; I=3/2+; mu/mu_N=+0.2705(8)
Au200_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=(1-)
Au201_SPIN_PROPERTIES        = SpinProperties(1.5, None, None)  # NUBASE2020; I=3/2+
Au202_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=(1-)
Au203_SPIN_PROPERTIES        = SpinProperties(1.5, None, None)  # NUBASE2020; I=3/2+
Au204_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=(2-)
Au205_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=3/2+#
Au206_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=6+#
Au207_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=3/2+#
Au208_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=6+#
Au209_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=3/2+#
Au210_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=6+#

# Pb
Pb185_SPIN_PROPERTIES        = SpinProperties(1.5, -1.137, -1.8486236356631145)  # IAEA; I=3/2-; mu/mu_N=-1.137(5)
Pb186_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Pb187_SPIN_PROPERTIES        = SpinProperties(1.5, -1.122, -1.84646737843508)  # IAEA; I=3/2-; mu/mu_N=-1.122(3)
Pb188_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Pb189_SPIN_PROPERTIES        = SpinProperties(1.5, -1.077, -1.8211976651200228)  # IAEA; I=3/2-; mu/mu_N=-1.077(9)
Pb190_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Pb191_SPIN_PROPERTIES        = SpinProperties(1.5, None, None)  # NUBASE2020; I=3/2-
Pb192_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Pb193_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=3/2-#
Pb194_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Pb195_SPIN_PROPERTIES        = SpinProperties(1.5, None, None)  # NUBASE2020; I=3/2-
Pb196_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Pb197_SPIN_PROPERTIES        = SpinProperties(1.5, -1.0718, -1.851804183317258)  # IAEA; I=3/2-; mu/mu_N=-1.0718(23)
Pb198_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Pb199_SPIN_PROPERTIES        = SpinProperties(1.5, -1.0707, -1.8595697062133563)  # IAEA; I=3/2-; mu/mu_N=-1.0707(14)
Pb200_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Pb201_SPIN_PROPERTIES        = SpinProperties(2.5, 0.6731, -0.6725171346099347)  # IAEA; I=5/2-; mu/mu_N=+0.6731(6)
Pb202_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Pb203_SPIN_PROPERTIES        = SpinProperties(2.5, 0.6841, -0.6638514842549722)  # IAEA; I=5/2-; mu/mu_N=+0.6841(6)
Pb204_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Pb205_SPIN_PROPERTIES        = SpinProperties(2.5, 0.7094, -0.6479823744896337)  # IAEA; I=5/2-; mu/mu_N=+0.7094(6)
Pb206_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Pb207_SPIN_PROPERTIES        = SpinProperties(0.5, 0.5906, 0.4796450904347988)  # IAEA; I=1/2-; mu/mu_N=+0.5906(4)
Pb208_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Pb209_SPIN_PROPERTIES        = SpinProperties(4.5, -1.4686, -1.4127748609421897)  # IAEA; I=9/2+; mu/mu_N=-1.4686(19)
Pb210_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Pb211_SPIN_PROPERTIES        = SpinProperties(4.5, -1.3991, -1.3970193088914251)  # IAEA; I=9/2+; mu/mu_N=-1.3991(12)
Pb212_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Pb213_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=(9/2+)
Pb214_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Pb215_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=9/2+#
Pb216_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Pb217_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=9/2+#
Pb218_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
Pb219_SPIN_PROPERTIES        = SpinProperties(None, None, None)  # NUBASE2020; I=11/2+#
Pb220_SPIN_PROPERTIES        = SpinProperties(0.0, 0.0, None)  # NUBASE2020; I=0+
