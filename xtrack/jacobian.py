import numpy as np
from .general import _print

from numpy.linalg import lstsq

class JacobianSolver:

    def __init__(self, func, maxsteps=20, tol=1e-20, n_bisections=8):
        self.func = func
        self.maxsteps = maxsteps
        self.tol = tol
        self.n_bisections = n_bisections

        self._penalty = None
        self._penalty_best = None
        self._xbest = None

    def _eval(self, x):
        y = self.func(x)
        penalty = np.dot(y, y)
        if penalty < self._penalty_best:
            self._penalty_best = penalty
            self._xbest = x.copy()
        return y, penalty

    def solve(self, x0):

        maskstart = None # Could be exposed to the user

        myf = self.func

        x0 = np.array(x0)
        x = x0.copy()
        if maskstart is None:
            maskstart = np.ones(len(myf(x0)), dtype=bool)
        mask = maskstart.copy()

        self._xbest = x0.copy()
        self._penalty_best = 1e200
        ncalls = 0
        info = {}
        for step in range(self.maxsteps):
            # start
            mask[:] = maskstart
            # test penalty
            y, penalty = self._eval(x) # will need to handle mask
            ncalls += 1
            if penalty < self.tol:
                _print("tolerance met")
                break
            # Equation search
            jac = myf.get_jacobian(x) # will need to handle mask
            ncalls += len(x)
            xstep = lstsq(jac, y, rcond=None)[0]  # newton step
            newpen = penalty * 2
            alpha = -1
            while newpen > penalty:  # bisec search
                if alpha > self.n_bisections:
                    break
                alpha += 1
                l = 2.0**-alpha

                y, newpen = self._eval(x - l * xstep)

                ncalls += 1
            x -= l * xstep  # update solution

            if alpha > 30:
                raise RuntimeError("not any progress")

        return self._xbest
