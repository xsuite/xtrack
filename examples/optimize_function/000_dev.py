import numpy as np
import xtrack as xt

class ActionCall(xt.Action):
    def __init__(self, function, vary):
        self.vary = vary
        self.function = function

    def run(self):
        x = [vv.container[vv.name] for vv in self.vary]
        return self.function(x)

    def get_targets(self, ftar):
        tars = []
        for ii in range(len(ftar)):
            tars.append(xt.Target(ii, ftar[ii], action=self))

        return tars

def opt_from_callable(function, x0, steps, tar, tols):
    x0 = np.array(x0)
    x = x0.copy()
    vary = [xt.Vary(ii, container=x, step=steps[ii]) for ii in range(len(x))]

    line = xt.Line() # dummy line to get the match (to be cleaned up)
    opt = line.match(
        solve=False,
        vary=vary,
        targets=ActionCall(function, vary).get_targets(tar)
    )

    for ii, tt in enumerate(opt.targets):
        tt.tol = tols[ii]

    return opt



def my_function(x):
    return [(x[0]-0.0001)**2, (x[1]-0.0003)**2, (x[2]+0.0005)**2]

x0 = [0., 0., 0.]

opt = opt_from_callable(my_function, x0=x0, steps=[1e-6, 1e-6, 1e-6],
                        tar=[0., 0., 0.], tols=[1e-12, 1e-12, 1e-12])
opt.solve()

