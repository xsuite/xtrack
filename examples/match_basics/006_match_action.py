import numpy as np
import xtrack as xt
import NAFFlib as nl



class ActionMeasAmplDet(xt.Action):

    def __init__(self, line, nemitt_x, nemitt_y, num_turns=256,
                 a0_sigmas=0.01, a1_sigmas=0.1, a2_sigmas=0.2):

        self.a0_sigmas = a0_sigmas
        self.a1_sigmas = a1_sigmas
        self.a2_sigmas = a2_sigmas

        self.line = line
        self.num_turns = num_turns
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y


    def run(self):

        a0_sigmas = self.a0_sigmas
        a1_sigmas = self.a1_sigmas
        a2_sigmas = self.a2_sigmas

        Jx_0 = a0_sigmas**2 * self.nemitt_x / 2
        Jx_1 = a1_sigmas**2 * self.nemitt_x / 2
        Jx_2 = a2_sigmas**2 * self.nemitt_x / 2
        Jy_0 = a0_sigmas**2 * self.nemitt_y / 2
        Jy_1 = a1_sigmas**2 * self.nemitt_y / 2
        Jy_2 = a2_sigmas**2 * self.nemitt_y / 2

        particles = self.line.build_particles(
                            method='4d',
                            zeta=0, delta=0,
                            x_norm=[a1_sigmas, a2_sigmas, a0_sigmas, a0_sigmas],
                            y_norm=[a0_sigmas, a0_sigmas, a1_sigmas, a2_sigmas],
                            nemitt_x=self.nemitt_x, nemitt_y=self.nemitt_y)

        self.line.track(particles,
                        num_turns=self.num_turns, time=True,
                        turn_by_turn_monitor=True)
        mon = line.record_last_track

        assert np.all(particles.state > 0)

        qx = np.zeros(4)
        qy = np.zeros(4)

        for ii in range(len(qx)):
            qx[ii] = nl.get_tune(mon.x[ii, :])
            qy[ii] = nl.get_tune(mon.y[ii, :])

        det_xx = (qx[1] - qx[0]) / (Jx_2 - Jx_1)
        det_yy = (qy[3] - qy[2]) / (Jy_2 - Jy_1)
        det_xy = (qx[3] - qx[2]) / (Jy_2 - Jy_1)
        det_yx = (qy[1] - qy[0]) / (Jx_2 - Jx_1)

        return {'det_xx': det_xx, 'det_yy': det_yy,
                'det_xy': det_xy, 'det_yx': det_yx}

# Load a line and build a tracker
line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.build_tracker()

nemitt_x = 2.5e-6
nemitt_y = 2.5e-6

action = ActionMeasAmplDet(line=line, nemitt_x=nemitt_x, nemitt_y=nemitt_y)

action.run()

opt = line.match(vary=xt.VaryList(['kof.a23b1', 'kod.a23b1'], step=1.),
                 targets=[action.target('det_xx', 1e3, tol=1e1),
                          action.target('det_yy', 2e3, tol=1e1)])
