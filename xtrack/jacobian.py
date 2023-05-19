import numpy as np
from .general import _print

from numpy.linalg import lstsq

class JacobianSolver:

    def __init__(self, func, limits, maxsteps=20, tol=1e-20, n_bisections=3,
                 n_no_improvement=5, verbose=False):
        self.func = func
        self.limits = limits
        self.maxsteps = maxsteps
        self.tol = tol
        self.n_bisections = n_bisections
        self.n_no_improvement = n_no_improvement

        self._penalty_best = None
        self._xbest = None
        self.verbose = verbose

    def _eval(self, x):
        y = self.func(x)
        penalty = np.sqrt(np.dot(y, y))
        if self.verbose:
            _print(f"penalty: {penalty}")
        if penalty < self._penalty_best:
            if self._penalty_best - penalty > 1e-20: #????????????
                self._step_best = self._step
            self._penalty_best = penalty
            self._xbest = x.copy()
            if self.verbose:
                _print(f"new best: {self._penalty_best}")
        return y, penalty

    def solve(self, x0):

        myf = self.func

        x0 = np.array(x0, dtype=float)
        x = x0.copy()

        self._xbest = x0.copy()
        self._penalty_best = 1e200
        ncalls = 0
        info = {}
        mask_for_next_step = np.ones(len(x0), dtype=bool)
        for step in range(self.maxsteps):

            self._step = step
            # test penalty
            y, penalty = self._eval(x) # will need to handle mask
            ncalls += 1
            if penalty < self.tol:
                if self.verbose:
                    _print("Jacobian tolerance met")
                break
            # Equation search
            jac = myf.get_jacobian(x) # will need to handle mask
            ncalls += len(x)

            # lstsq using only the the variables that were not at the limit
            # in the previous step
            xstep = np.zeros(len(x))

            xstep[mask_for_next_step] = lstsq(jac[:, mask_for_next_step], y, rcond=None)[0]  # newton step
            mask_for_next_step[:] = True
            self._last_xstep = xstep.copy()

            newpen = penalty * 2
            alpha = -1

            while newpen > penalty:  # bisec search
                if alpha > self.n_bisections:
                    break
                alpha += 1
                l = 2.0**-alpha
                if self.verbose:
                    _print(f"\n--> step {step} alpha {alpha}\n")

                this_xstep = l * xstep
                mask_hit_limit = np.zeros(len(x), dtype=bool)
                for ii in range(len(x)):
                    if x[ii] - this_xstep[ii] < self.limits[ii][0]:
                        this_xstep[ii] = 0
                        mask_hit_limit[ii] = True
                    elif x[ii] - this_xstep[ii] > self.limits[ii][1]:
                        this_xstep[ii] = 0
                        mask_hit_limit[ii] = True

                y, newpen = self._eval(x - this_xstep) # will need to handle mask

                ncalls += 1
            x -= this_xstep  # update solution
            mask_for_next_step = ~mask_hit_limit

            if self.verbose:
                _print(f"step {step} step_best {self._step_best}")
            if step > self._step_best + self.n_no_improvement:
                if self.verbose:
                    _print("No progress, stopping")
                break
        else:
            if self.verbose:
                _print("Max steps reached")

        return self._xbest
