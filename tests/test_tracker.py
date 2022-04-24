import numpy as np
import xobjects as xo
import xtrack as xt
import xpart as xp
import xfields as xf


def test_ebe_monitor():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        line = xt.Line(elements=[xt.Multipole(knl=[0, 1.]),
                                xt.Drift(length=0.5),
                                xt.Multipole(knl=[0, -1]),
                                xt.Cavity(frequency=400e7, voltage=6e6),
                                xt.Drift(length=.5),
                                xt.Drift(length=0)])

        tracker = line.build_tracker(_context=context)

        particles = xp.Particles(x=[1e-3, -2e-3, 5e-3], y=[2e-3, -4e-3, 3e-3],
                                zeta=1e-2, p0c=7e12, mass0=xp.PROTON_MASS_EV,
                                _context=context)

        tracker.track(particles.copy(), turn_by_turn_monitor='ONE_TURN_EBE')

        mon = tracker.record_last_track

        for ii, ee in enumerate(line.elements):
            for tt, nn in particles._structure['per_particle_vars']:
                assert np.all(particles.to_dict()[nn] == getattr(mon, nn)[:, ii])
            ee.track(particles)
            particles.at_element += 1

def test_cycle():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        d0 = xt.Drift()
        c0 = xt.Cavity()
        d1 = xt.Drift()
        r0 = xt.SRotation()


        for collective in [True, False]:
            line = xt.Line(elements=[d0, c0, d1, r0])
            d1.iscollective = collective

            tracker = xt.Tracker(line=line, _context=context)

            ctracker_name = tracker.cycle(name_first_element='e2')
            ctracker_index = tracker.cycle(index_first_element=2)

            for ctracker in [ctracker_index, ctracker_name]:
                assert ctracker.line.element_names[0] == 'e2'
                assert ctracker.line.element_names[1] == 'e3'
                assert ctracker.line.element_names[2] == 'e0'
                assert ctracker.line.element_names[3] == 'e1'

                assert ctracker.line.elements[0] is d1
                assert ctracker.line.elements[1] is r0
                assert ctracker.line.elements[2] is d0
                assert ctracker.line.elements[3] is c0

def test_partial_tracking():
    line = xt.Line(elements=[xt.Multipole(knl=[0, 1.]),
                                xt.Drift(length=0.5),
                                xt.Multipole(knl=[0, -1]),
                                xt.Cavity(frequency=400e7, voltage=6e6),
                                xt.Drift(length=.5),
                                xt.Drift(length=0)])
    n_elem = len(line.element_names)
    tracker = line.build_tracker()
    particles_init = xp.Particles(x=[1e-3, -2e-3, 5e-3], y=[2e-3, -4e-3, 3e-3],
                                zeta=1e-2, p0c=7e12, mass0=xp.PROTON_MASS_EV,
                                at_turn=0, at_element=0)

    _default_track(tracker, particles_init)
    _ele_start_until_end(tracker, particles_init)
    _ele_start_with_shift(tracker, particles_init)
    _ele_start_with_shift_more_turns(tracker, particles_init)
    _ele_stop_from_start(tracker, particles_init)
    _ele_start_to_ele_stop(tracker, particles_init)
    _ele_start_to_ele_stop_with_overflow(tracker, particles_init)


def test_partial_tracking_with_collective():
    k2engine = xc.K2Engine(100)
    line = xt.Line(elements=[xt.Multipole(knl=[0, 1.]),
                                xt.Drift(length=0.5),
                                xt.Drift(length=0.75),
                                xc.K2Collimator(k2engine=k2engine, active_length=0.6, angle=90,
                                               inactive_front=0, inactive_back=0, material='MoGR', is_active=False),
                                xt.Multipole(knl=[0, -1]),
                                xt.Cavity(frequency=400e7, voltage=6e6),
                                xt.Drift(length=.5),
                                xc.K2Collimator(k2engine=k2engine, active_length=0.6, angle=0,
                                               inactive_front=0, inactive_back=0, material='MoGR', is_active=False),
                                xt.Drift(length=0)])
    n_elem = len(line.element_names)
    tracker = line.build_tracker()
    particles_init = xp.Particles(x=[1e-3, -2e-3, 5e-3], y=[2e-3, -4e-3, 3e-3],
                                zeta=1e-2, p0c=7e12, mass0=xp.PROTON_MASS_EV,
                                at_turn=0, at_element=0)

    _default_track(tracker, particles_init)
    _ele_start_until_end(tracker, particles_init)
    _ele_start_with_shift(tracker, particles_init)
    _ele_start_with_shift_more_turns(tracker, particles_init)
    _ele_stop_from_start(tracker, particles_init)
    _ele_start_to_ele_stop(tracker, particles_init)
    _ele_start_to_ele_stop_with_overflow(tracker, particles_init)


# Track, from any ele_start, until the end of the first, second, and tenth turn
def _default_track(tracker, particles_init):
    n_elem = len(tracker.line.element_names)
    for turns in [1, 2, 10]:
        expected_end_turn = turns
        expected_end_element = 0

        particles = particles_init.copy()
        tracker.track(particles, num_turns=turns)
        check, end_turn, end_element = _get_at_turn_element(particles)
        assert check and end_turn==expected_end_turn and end_element==expected_end_element

# Track, from any ele_start, until the end of the first, second, and tenth turn
def _ele_start_until_end(tracker, particles_init):
    n_elem = len(tracker.line.element_names)
    for turns in [1, 2, 10]:
        for start in range(n_elem):
            expected_end_turn = turns
            expected_end_element = 0

            particles = particles_init.copy()
            tracker.track(particles, num_turns=turns, ele_start=start)
            check, end_turn, end_element = _get_at_turn_element(particles)
            assert check and end_turn==expected_end_turn and end_element==expected_end_element

# Track, from any ele_start, any shifts that stay within the first turn
def _ele_start_with_shift(tracker, particles_init):
    n_elem = len(tracker.line.element_names)
    for start in range(n_elem):
        for shift in range(1,n_elem-start):
            expected_end_turn = 0
            expected_end_element = start+shift

            particles = particles_init.copy()
            tracker.track(particles, ele_start=start, num_elements=shift)
            check, end_turn, end_element = _get_at_turn_element(particles)
            assert check and end_turn==expected_end_turn and end_element==expected_end_element

# Track, from any ele_start, any shifts that are larger than one turn (up to 3 turns)
def _ele_start_with_shift_more_turns(tracker, particles_init):
    n_elem = len(tracker.line.element_names)
    for start in range(n_elem):
        for shift in range(n_elem-start, 3*n_elem+1):
            expected_end_turn = round(np.floor( (start+shift)/n_elem ))
            expected_end_element = start + shift - n_elem*expected_end_turn

            particles = particles_init.copy()
            tracker.track(particles, ele_start=start, num_elements=shift)
            check, end_turn, end_element = _get_at_turn_element(particles)
            assert check and end_turn==expected_end_turn and end_element==expected_end_element

# Track from the start until any ele_stop in the first, second, and tenth turn
def _ele_stop_from_start(tracker, particles_init):
    n_elem = len(tracker.line.element_names)
    for turns in [1, 2, 10]:
        for stop in range(1, n_elem):
            expected_end_turn = turns-1
            expected_end_element = stop

            particles = particles_init.copy()
            tracker.track(particles, num_turns=turns, ele_stop=stop)
            check, end_turn, end_element = _get_at_turn_element(particles)
            assert check and end_turn==expected_end_turn and end_element==expected_end_element

# Track from any ele_start until any ele_stop that is larger than ele_start (so no overflow)
# for one, two, and ten turns
def _ele_start_to_ele_stop(tracker, particles_init): 
    n_elem = len(tracker.line.element_names)
    for turns in [1, 2, 10]:
        for start in range(n_elem):
            for stop in range(start+1,n_elem):
                expected_end_turn = turns-1
                expected_end_element = stop

                particles = particles_init.copy()
                tracker.track(particles, num_turns=turns, ele_start=start, ele_stop=stop)
                check, end_turn, end_element = _get_at_turn_element(particles)
                assert check and end_turn==expected_end_turn and end_element==expected_end_element

# Track from any ele_start until any ele_stop that is smaller than or equal to ele_start (turn overflow)
# for one, two, and ten turns
def _ele_start_to_ele_stop_with_overflow(tracker, particles_init):
    n_elem = len(tracker.line.element_names)
    for turns in [1, 2, 10]:
        for start in range(n_elem):
            for stop in range(start+1):
                expected_end_turn = turns
                expected_end_element = stop

                particles = particles_init.copy()
                tracker.track(particles, num_turns=turns, ele_start=start, ele_stop=stop)
                check, end_turn, end_element = _get_at_turn_element(particles)
                assert check and end_turn==expected_end_turn and end_element==expected_end_element


# Quick helper function to:
#   1) check that all survived particles are at the same element and turn
#   2) return that element and turn
def _get_at_turn_element(particles):
    at_element = np.unique(particles.at_element[particles.state>0])
    at_turn = np.unique(particles.at_turn[particles.state>0])
    all_together = len(at_turn)==1 and len(at_element)==1
    return all_together, at_turn[0], at_element[0]

