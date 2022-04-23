import numpy as np
import xobjects as xo
import xtrack as xt
import xpart as xp

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
 
    tracker = line.build_tracker()
    particles_init = xp.Particles(x=[1e-3, -2e-3, 5e-3], y=[2e-3, -4e-3, 3e-3],
                                zeta=1e-2, p0c=7e12, mass0=xp.PROTON_MASS_EV,
                                at_turn=0, at_element=0)

    # Track two elements from lattice start
    particles = particles_init.copy()
    tracker.track(particles, num_turns=1, num_elements=2)
    at_element = np.unique(particles.at_element[particles.state>0])
    at_turn = np.unique(particles.at_turn[particles.state>0])
    assert len(at_turn)==1 and at_turn[0]==0 and \
            len(at_element)==1 and at_element[0]==2

    # Track two elements, starting from first element
    particles = particles_init.copy()
    tracker.track(particles, num_turns=1, ele_start=0, num_elements=2)
    at_element = np.unique(particles.at_element[particles.state>0])
    at_turn = np.unique(particles.at_turn[particles.state>0])
    assert len(at_turn)==1 and at_turn[0]==0 and \
            len(at_element)==1 and at_element[0]==2

    # Track two elements, starting from third element
    particles = particles_init.copy()
    tracker.track(particles, num_turns=1, ele_start=2, num_elements=2)
    at_element = np.unique(particles.at_element[particles.state>0])
    at_turn = np.unique(particles.at_turn[particles.state>0])
    return at_element, at_turn
    assert len(at_turn)==1 and at_turn[0]==0 and \
            len(at_element)==1 and at_element[0]==4

    # Track from lattice start until third element
    particles = particles_init.copy()
    tracker.track(particles, num_turns=1, ele_stop=2)
    at_element = np.unique(particles.at_element[particles.state>0])
    at_turn = np.unique(particles.at_turn[particles.state>0])
    assert len(at_turn)==1 and at_turn[0]==0 and \
            len(at_element)==1 and at_element[0]==2
    
    
    
    
    
