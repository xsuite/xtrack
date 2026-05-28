import numpy as np

from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField


class TiltedSolenoid:

    def __init__(self, L, a, B0, theta):

        self._sf = SolenoidField(L=L, a=a, B0=B0, z0=0)
        self.theta = theta
        self._break = False

    def get_field(self, x, y, z):

        if self._break:
            breakpoint()

        theta = self.theta
        stheta = np.sin(theta)
        ctheta = np.cos(theta)
        x_sol = x * ctheta + z * stheta
        z_sol = -x * stheta + z * ctheta
        y_sol = y

        bx_sol, by_sol, bz_sol = self._sf.get_field(x_sol, y_sol, z_sol)

        bx = bx_sol * np.cos(theta) - bz_sol * np.sin(theta)
        bz = bx_sol * np.sin(theta) + bz_sol * np.cos(theta)
        by = by_sol

        return bx, by, bz

    def compute_pure_field_derivatives(
            self, s, direction, step, component, max_order=4, min_order=1):
        return SolenoidField.compute_pure_field_derivatives(
            self, s=s, direction=direction, step=step, component=component,
            max_order=max_order, min_order=min_order)

    def compute_pure_by_derivatives(self, s, direction, step, max_order=4):
        return self.compute_pure_field_derivatives(
            s=s, direction=direction, step=step,
            component='y', max_order=max_order, min_order=1)
