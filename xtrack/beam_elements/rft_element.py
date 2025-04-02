"""
RF-Track <-> Xsuite interface

Authors: Andrea Latina, Gianni Iadarola
Date: 22.11.2023

"""

import numpy as np

class RFT_Element:
    """Beam element calling RF-Track.

    Xtrack will use RF-Track to track a bunch through an RF-Track element. This can
    include any RF-Track element, e.g., field maps, absorbers, travelling- and standing-
    wave structures, electron coolers and others.

    Parameters:
        - element: An RF-Track element.
    """

    # init xtrack variables
    iscollective = True # <-- To state that the element uses a python track method
    isthick = True

    def __init__(self, element):

        import RF_Track as rft
        self.length = element.get_length()
        self.lattice = rft.Lattice()
        self.lattice.append_ref(element)
        self.bunch_in = rft.Bunch6d()
        self.bunch_out = rft.Bunch6d()
        self.arr_for_rft = np.empty(0)
        self.arr_for_xt = np.empty(0)

    def track(self, particles, increment_at_element=False):

        p = particles
        if p.x.shape[0]==0:
           return

        # Prepare input for RF-Track
        self.p = 1. + p.delta
        self.pz = np.sqrt(self.p**2 - p.px**2 - p.py**2)
        self.arr_for_rft.resize((len(p.x), 10))
        self.arr_for_rft[:,0] = p.x * 1e3 # mm
        self.arr_for_rft[:,1] = p.px * 1e3 / self.pz # mrad, xp
        self.arr_for_rft[:,2] = p.y * 1e3 # mm
        self.arr_for_rft[:,3] = p.py * 1e3 / self.pz # mrad, yp
        self.arr_for_rft[:,4] = -p.zeta * 1e3 / p.beta0 # mm/c, t
        self.arr_for_rft[:,5] = p.p0c * self.p / 1e6 # MeV
        self.arr_for_rft[:,6] = p.mass0 / 1e6 # MeV
        self.arr_for_rft[:,7] = p.q0
        self.arr_for_rft[:,8] = p.weight
        self.arr_for_rft[p.state==1,9] = np.nan # nan == not lost

        # Track the refernece particle
        import RF_Track as rft
        pref0 = rft.Bunch6d(np.array([0,0,0,0,0,p.p0c[0]/1e6,p.mass0/1e6,p.q0,p.weight.sum()]))
        pref1 = self.lattice.track(pref0)

        # Track the bunch
        self.bunch_in.set_phase_space(self.arr_for_rft)
        self.bunch_out = self.lattice.track(self.bunch_in)
        self.arr_for_xt = self.bunch_out.get_phase_space('%x %Px %y %Py %t %Pc %lost %id', 'all')

        # Update particles
        self.arr_for_xt = self.arr_for_xt[self.arr_for_xt[:,7].argsort()] # sort by particle id
        p.x  = self.arr_for_xt[:,0] / 1e3 # m
        p.px = self.arr_for_xt[:,1] * 1e6 / p.p0c # rad
        p.y  = self.arr_for_xt[:,2] / 1e3 # m
        p.py = self.arr_for_xt[:,3] * 1e6 / p.p0c # rad
        p.zeta += self.length - self.arr_for_xt[:,4] * p.beta0 / 1e3 # m
        p.delta = (self.arr_for_xt[:,5] * 1e6 - p.p0c) / p.p0c #
        p.state[np.isnan(self.arr_for_xt[:,6])!=np.isnan(self.arr_for_rft[:,9])] = -400 # lost in RF-Track
