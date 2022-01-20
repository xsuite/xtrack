from ..mathlibs import MathlibDefault


class QGauss(object):
    @staticmethod
    def calc_cq(q, mathlib=MathlibDefault, EPS=1e-6):
        assert q < 3
        cq = mathlib.sqrt(mathlib.pi)
        if q >= (1 + EPS):
            cq *= mathlib.gamma((3 - q) / (2 * q - 2))
            cq /= mathlib.sqrt((q - 1)) * mathlib.gamma(1 / (q - 1))
        elif q <= (1 - EPS):
            cq *= 2 * mathlib.gamma(1 / (1 - q))
            cq /= (
                (3 - q)
                * mathlib.sqrt(1 - q)
                * mathlib.gamma((3 - q) / (2 - 2 * q))
            )
        return cq

    @staticmethod
    def sqrt_beta(sigma, mathlib=MathlibDefault):
        assert sigma > 0
        return 1 / (mathlib.sqrt(2) * sigma)

    @staticmethod
    def exp_q(x, q, mathlib=MathlibDefault, EPS=1e-6):
        assert q < 3
        if mathlib.abs(1 - q) > EPS:
            u_plus = 1 + x * (1 - q)
            if u_plus < 0:
                u_plus = 0
            return mathlib.pow(u_plus, 1 / (1 - q))
        else:
            return mathlib.exp(x)

    def __init__(self, q=1.0, mathlib=MathlibDefault, cq_eps=1e-6):
        assert q < 3
        self._m = mathlib
        self._q = q
        self._cq = QGauss.calc_cq(q, mathlib=mathlib, EPS=cq_eps)

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q_value):
        assert q_value < 3
        self._q = q_value
        self._cq = QGauss.calc_cq(self._q, self._m)

    @property
    def cq(self):
        return self._cq

    def min_support(self, sqrt_beta):
        assert self._q < 3
        assert sqrt_beta > 0
        if self._q >= 1:
            return -1e10
        else:
            return -1 / self._m.sqrt(sqrt_beta * sqrt_beta * (1 - self._q))

    def max_support(self, sqrt_beta):
        return -(self.min_support(sqrt_beta))

    def eval(self, x, sqrt_beta, mu=0.0):
        assert self._m.abs(self._cq) > 0
        assert self._q < 3
        assert sqrt_beta > 0
        factor = sqrt_beta / self._cq
        arg = sqrt_beta * sqrt_beta
        arg *= (x - mu) * (x - mu)
        return factor * QGauss.exp_q(-arg, self._q, self._m)
