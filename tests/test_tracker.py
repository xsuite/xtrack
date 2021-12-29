import xobjects as xo
import xtrack as xt

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