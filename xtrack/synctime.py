import xobjects as xo

COAST_STATE_RANGE_START= 1000000
DEFAULT_FRAME_RELATIVE_LENGTH = 0.9

class SyncTime:

    def __init__(self, circumference, id, frame_relative_length=None,
                 at_start=False, at_end=False):
        if frame_relative_length is None:
            frame_relative_length = DEFAULT_FRAME_RELATIVE_LENGTH
        assert id > COAST_STATE_RANGE_START
        self.id = id
        self.frame_relative_length = frame_relative_length
        self.circumference = circumference
        self.at_start = at_start
        self.at_end = at_end

    def track(self, particles):

        assert isinstance(particles._context, xo.ContextCpu), (
                          'SyncTime only available for CPU for now')

        beta0 = particles._xobject.beta0[0]
        beta1 = beta0 / self.frame_relative_length
        beta0_beta1 = beta0 / beta1

        mask_alive = particles.state > 0

        zeta_min = -self.circumference/ 2 * beta0_beta1 + particles.s * (
                   1 - beta0_beta1)

        if (self.at_start and particles.at_turn[0] == 0
                and not (particles.state == -COAST_STATE_RANGE_START).any()): # done by the user
            mask_stop = mask_alive * (particles.zeta < zeta_min)
            particles.state[mask_stop] = -COAST_STATE_RANGE_START
            particles.zeta[mask_stop] += self.circumference * beta0 / beta1

        # Resume particles previously stopped
        particles.state[particles.state==-self.id] = 1
        particles.reorganize()

        # Identify particles that need to be stopped
        zeta_min = -self.circumference/ 2 * beta0_beta1 + particles.s * (1 - beta0_beta1)
        mask_stop = mask_alive & (particles.zeta < zeta_min)

        # Check if some particles are too fast
        mask_too_fast = mask_alive & (
            particles.zeta > zeta_min + self.circumference * beta0_beta1)
        if mask_too_fast.any():
            raise ValueError('Some particles move faster than the time window')

        # For debugging (expected to be triggered)
        # mask_out_of_circumference = mask_alive & (
        #       (particles.zeta > self.circumference / 2)
        #     | (particles.zeta < -self.circumference / 2))
        # if mask_out_of_circumference.any():
        #     raise ValueError('Some particles are out of the circumference')

        # Update zeta for particles that are stopped
        particles.zeta[mask_stop] += beta0_beta1 * self.circumference

        # Stop particles
        particles.state[mask_stop] = -self.id

        if self.at_end:
            mask_alive = particles.state > 0
            particles.zeta[mask_alive] -= (
                self.circumference * (1 - beta0_beta1))

        if self.at_end and particles.at_turn[0] == 0:
            particles.state[particles.state==-COAST_STATE_RANGE_START] = 1

def install_sync_time_at_collective_elements(line, frame_relative_length=None):

    circumference = line.get_length()

    ltab = line.get_table()
    tab_collective = ltab.rows[ltab.iscollective]
    for ii, nn in enumerate(tab_collective.name):
        cc = x=SyncTime(circumference=circumference,
                        frame_relative_length=frame_relative_length,
                        id=COAST_STATE_RANGE_START + ii + 1)
        line.insert_element(element=cc, name=f'synctime_{ii}', at=nn)

    synctime_start = SyncTime(circumference=circumference,
                        frame_relative_length=frame_relative_length,
                        id=COAST_STATE_RANGE_START + len(tab_collective)+1,
                        at_start=True)
    synctime_end = SyncTime(circumference=circumference,
                        frame_relative_length=frame_relative_length,
                        id=COAST_STATE_RANGE_START + len(tab_collective)+2,
                        at_end=True)

    line.insert_element(element=synctime_start, name='synctime_start', at_s=0)
    line.append_element(synctime_end, name='synctime_end')

def prepare_particles_for_sync_time(particles, line):
    synctime_start = line['synctime_start']
    beta0 = particles._xobject.beta0[0]
    beta1 = beta0 / synctime_start.frame_relative_length
    beta0_beta1 = beta0 / beta1
    zeta_min = -synctime_start.circumference/ 2 * beta0_beta1 + particles.s * (
                1 - beta0_beta1)
    mask_alive = particles.state > 0
    mask_stop = mask_alive * (particles.zeta < zeta_min)
    particles.state[mask_stop] = -COAST_STATE_RANGE_START
    particles.zeta[mask_stop] += synctime_start.circumference * beta0 / beta1