from scipy.special import wofz
from scipy.special import gamma as tgamma


class MathlibDefault:

    from numpy import sqrt, exp, sin, cos, abs, pi, tan, interp, linspace
    from numpy import power as pow

    @classmethod
    def wfun(cls, z_re, z_im):
        w = wofz(z_re + 1j * z_im)
        return w.real, w.imag

    @classmethod
    def gamma(cls, arg):
        assert arg > 0.0
        return tgamma(arg)
