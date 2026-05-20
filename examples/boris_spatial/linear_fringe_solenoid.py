import numpy as np

class LinearFringeSolenoid:

    def __init__(self, B0, s1, s2, s3, s4):
        self.B0 = B0
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4

    def get_field(self, x, y, s):

        mask_entry_taper = (s >= self.s1) & (s < self.s2)
        mask_constant = (s >= self.s2) & (s < self.s3)
        mask_exit_taper = (s >= self.s3) & (s < self.s4)

        Bx = np.zeros_like(x)
        By = np.zeros_like(y)
        Bz = np.zeros_like(s)

        Bz[mask_entry_taper] = self.B0 * (s[mask_entry_taper] - self.s1) / (self.s2 - self.s1)
        Bz[mask_constant] = self.B0
        Bz[mask_exit_taper] = self.B0 * (self.s4 - s[mask_exit_taper]) / (self.s4 - self.s3)

        # Bx = -x dBz/dz /2
        # By = -y dBz/dz /2

        dBz_ds_entry = self.B0 / (self.s2 - self.s1)
        dBz_ds_exit = -self.B0 / (self.s4 - self.s3)

        Bx[mask_entry_taper] = -x[mask_entry_taper] * dBz_ds_entry / 2
        Bx[mask_exit_taper] = -x[mask_exit_taper] * dBz_ds_exit / 2
        By[mask_entry_taper] = -y[mask_entry_taper] * dBz_ds_entry / 2
        By[mask_exit_taper] = -y[mask_exit_taper] * dBz_ds_exit / 2

        return Bx, By, Bz