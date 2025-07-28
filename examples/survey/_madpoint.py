import numpy as np

class MadPoint(object):
    @classmethod
    def from_survey(cls, name, mad=None, xsuite_survey=None):
        return cls(name, mad=mad, use_twiss=False, use_survey=True,
                   xsuite_survey=xsuite_survey)

    @classmethod
    def from_twiss(cls, name, mad):
        return cls(name, mad, use_twiss=True, use_survey=False)

    def __init__(self, name, mad=None, use_twiss=True, use_survey=True,
                 xsuite_twiss=None, xsuite_survey=None):

        self.use_twiss = use_twiss
        self.use_survey = use_survey

        if not (use_survey) and not (use_twiss):
            raise ValueError(
                "use_survey and use_twiss cannot be False at the same time"
            )

        self.tx = None
        self.ty = None
        self.tpx = None
        self.tpy = None

        self.sx = None
        self.sy = None
        self.sz = None
        self.sp = None
        theta = 0.0
        phi = 0.0
        psi = 0.0

        if mad is not None:

            self.name = name
            if use_twiss:
                assert xsuite_survey is None
                twiss = mad.table.twiss
                names = twiss.name
            if use_survey:
                assert xsuite_twiss is None
                survey = mad.table.survey
                names = survey.name
                # patch for this issue https://github.com/hibtc/cpymad/issues/91 
                for ii, nn in enumerate(names):
                    if not nn.endswith(':1'):
                        names[ii] = nn+':1'

            idx = np.where(names == name)[0][0]

            if use_twiss:
                self.tx = twiss.x[idx]
                self.ty = twiss.y[idx]
                self.tpx = twiss.px[idx]
                self.tpy = twiss.py[idx]

            if use_survey:
                self.sx = survey.x[idx]
                self.sy = survey.y[idx]
                self.sz = survey.z[idx]
                self.sp = np.array([self.sx, self.sy, self.sz])
                theta = survey.theta[idx]
                phi = survey.phi[idx]
                psi = survey.psi[idx]
        else:


            if use_twiss:
                assert xsuite_twiss is not None
                idx = np.where(np.array(xsuite_twiss['name']) == name)[0][0]
                self.tx = xsuite_twiss.x[idx]
                self.ty = xsuite_twiss.y[idx]
                self.tpx = xsuite_twiss.px[idx]
                self.tpy = xsuite_twiss.py[idx]

            if use_survey:
                assert xsuite_survey is not None
                idx = np.where(np.array(xsuite_survey['name']) == name)[0][0]
                self.sx = xsuite_survey.X[idx]
                self.sy = xsuite_survey.Y[idx]
                self.sz = xsuite_survey.Z[idx]
                self.sp = np.array([self.sx, self.sy, self.sz])
                theta = xsuite_survey.theta[idx]
                phi = xsuite_survey.phi[idx]
                psi = xsuite_survey.psi[idx]

        thetam = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )
        phim = np.array(
            [
                [1, 0, 0],
                [0, np.cos(phi), np.sin(phi)],
                [0, -np.sin(phi), np.cos(phi)],
            ]
        )
        psim = np.array(
            [
                [np.cos(psi), -np.sin(psi), 0],
                [np.sin(psi), np.cos(psi), 0],
                [0, 0, 1],
            ]
        )
        wm = np.dot(thetam, np.dot(phim, psim))
        self.ex = np.dot(wm, np.array([1, 0, 0]))
        self.ey = np.dot(wm, np.array([0, 1, 0]))
        self.ez = np.dot(wm, np.array([0, 0, 1]))

        self.p = np.array([0.0, 0.0, 0.0])

        if use_twiss:
            self.p += self.ex * self.tx + self.ey * self.ty

        if use_survey:
            self.p += self.sp

    def shift_survey(self, delta):
        self.sx -= delta[0]
        self.sy -= delta[1]
        self.sz -= delta[2]
        self.sp -= delta
        self.p -= delta

    def dist(self, other):
        return np.sqrt(np.sum((self.p - other.p) ** 2))

    def distxy(self, other):
        dd = self.p - other.p
        return np.dot(dd, self.ex), np.dot(dd, self.ey)
