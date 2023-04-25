import numpy as np

import xtrack as xt

from . import lumi


class TargetLuminosity(xt.Target):

    def __init__(self, ip_name, luminosity, tol, num_colliding_bunches,
                 num_particles_per_bunch, nemitt_x, nemitt_y, sigma_z, f_rev,
                 crab=None):

        xt.Target.__init__(self, self.compute_luminosity, luminosity, tol=tol)

        self.ip_name = ip_name
        self.num_colliding_bunches = num_colliding_bunches
        self.num_particles_per_bunch = num_particles_per_bunch
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y
        self.sigma_z = sigma_z
        self.f_rev = f_rev
        self.crab = crab

    @property
    def scale(self):
        if self._scale is None:
            return 1/self.value

    @scale.setter
    def scale(self, value):
        self._scale = value

    def compute_luminosity(self, tw):
        assert len(tw._line_names) == 2
        return lumi.luminosity_from_twiss(
            n_colliding_bunches=self.num_colliding_bunches,
            num_particles_per_bunch=self.num_particles_per_bunch,
            ip_name=self.ip_name,
            nemitt_x=self.nemitt_x,
            nemitt_y=self.nemitt_y,
            sigma_z=self.sigma_z,
            twiss_b1=tw[tw._line_names[0]],
            twiss_b2=tw[tw._line_names[1]],
            f_rev=self.f_rev,
            crab=self.crab)

class TargetSeparationOrthogonalToCrossing(xt.Target):

    def __init__(self, ip_name):
        xt.Target.__init__(self, tar=self.projection, value=0, tol=1e-6, scale=1)
        self.ip_name = ip_name

    def projection(self, tw):
        assert len(tw._line_names) == 2
        ip_name = self.ip_name

        twb1 = tw[tw._line_names[0]]
        twb2 = tw[tw._line_names[1]].reverse()

        diff_px = twb1['px', ip_name] - twb2['px', ip_name]
        diff_py = twb1['py', ip_name] - twb2['py', ip_name]

        diff_p_mod = np.sqrt(diff_px**2 + diff_py**2)

        diff_x = twb1['x', ip_name] - twb2['x', ip_name]
        diff_y = twb1['y', ip_name] - twb2['y', ip_name]

        diff_r_mod = np.sqrt(diff_x**2 + diff_y**2)

        return (diff_x*diff_px + diff_y*diff_py)/(diff_p_mod*diff_r_mod)

class TargetSeparation(xt.Target):

    def __init__(self, ip_name, separation=None, separation_norm=None,
                 plane=None, nemitt_x=None, nemitt_y=None, tol=None, scale=None):


        # For now nemitt is a scalar, we can move to a tuple for different
        # emittances for the two beams

        if separation is None and separation_norm is None:
            raise ValueError('Either separation or separation_norm must be '
                             'provided')

        if separation is not None:
            value = separation
        elif separation_norm is not None:
            value = separation_norm

        if separation_norm is not None and (nemitt_x is None and nemitt_y is None):
            raise ValueError('nemitt must be provided if separation_norm is '
                             'provided')

        if plane is None: # In the future plane=None could be used to set the distance between 
                          # the two beams computed as sqrt(dx**2 + dy**2)
            raise ValueError('plane must be provided')
        assert plane in ['x', 'y']

        xt.Target.__init__(self, tar=self.get_separation, value=value, tol=tol,
                            scale=scale)

        self.ip_name = ip_name
        self.separation = separation
        self.separation_norm = separation_norm
        self.plane = plane
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y

    def get_separation(self, tw):

        assert len(tw._line_names) == 2
        tw1 = tw[tw._line_names[0]]
        tw2 = tw[tw._line_names[1]].reverse()

        if self.separation is not None:
            return np.abs(tw1[self.plane, self.ip_name] - tw2[self.plane, self.ip_name])
        elif self.separation_norm is not None:
            nemitt = self.nemitt_x if self.plane == 'x' else self.nemitt_y
            sigma1 = np.sqrt(
                tw1['bet'+self.plane, self.ip_name] * nemitt
                / tw1.particle_on_co.gamma0[0] / tw1.particle_on_co.beta0[0])
            sigma2 = np.sqrt(
                tw2['bet'+self.plane, self.ip_name] * nemitt
                / tw2.particle_on_co.gamma0[0] / tw2.particle_on_co.beta0[0])

            sigma = np.sqrt(sigma1*sigma2) # geometric mean of sigmas
            sep_norm = np.abs((tw1[self.plane, self.ip_name] - tw2[self.plane, self.ip_name]
                        )) / sigma

            return sep_norm