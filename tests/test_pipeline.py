import numpy as np
import pandas as pd

import xtrack as xt
import xpart as xp
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

from xtrack import PipelineStatus


@for_all_test_contexts
def test_multitracker(test_context):
    class DummyPipelinedElement:
        def __init__(self, n_hold=0):
            self.n_hold = n_hold
            self.iscollective = True
            self.i_hold = 0

        def track(self, particles):
            self.i_hold += 1
            if self.i_hold < self.n_hold:
                return PipelineStatus(on_hold=True,
                    info=f'stopped by internal counter {self.i_hold}/{self.n_hold}')
            else:
                self.i_hold = 0

    # We build two trackers, each with an element that puts the simulation on hold

    line1 = xt.Line(elements=[xt.Drift(length=1),
                            DummyPipelinedElement(n_hold=2),
                            xt.Drift(length=1)],
                    element_names=['d11', 'pipelnd_el1', 'd12'])
    line1.build_tracker(_context=test_context, enable_pipeline_hold=True)
    line1.reset_s_at_end_turn = False

    line2 = xt.Line(elements=[xt.Drift(length=1),
                            DummyPipelinedElement(n_hold=3),
                            xt.Drift(length=1)],
                    element_names=['d11', 'pipelnd_el2', 'd12'])
    line2.build_tracker(_context=test_context, enable_pipeline_hold=True)
    line2.reset_s_at_end_turn = False

    # We use a multitracker to track one two particles with the first tracker
    # and one particle with the second one

    p1 = xp.Particles(p0c=7e12, x=[0,0,0], _context=test_context)
    p2 = xp.Particles(p0c=7e12, x=[0,0,0], _context=test_context)
    p3 = xp.Particles(p0c=7e12, x=[0,0,0], _context=test_context)

    multitracker = xt.PipelineMultiTracker(
        branches=[xt.PipelineBranch(line=line1, particles=p1),
                xt.PipelineBranch(line=line1, particles=p2),
                xt.PipelineBranch(line=line2, particles=p3),
                ],
        enable_debug_log=True)

    multitracker.track(num_turns=4)

    # The simulation switched multiple times from one brach to the other (due to the
    # pipelined elements as it can be seen by the debug log (it is convenient to use
    # a pandas dataframe to view the debug log).

    log_df = pd.DataFrame(multitracker.debug_log)

    # Contains
    #     branch  track_session_turn held_by_element                             info
    # 0        0                   0     pipelnd_el1  stopped by internal counter 1/2
    # 1        1                   1     pipelnd_el1  stopped by internal counter 1/2
    # 2        2                   0     pipelnd_el2  stopped by internal counter 1/3
    # 3        0                   1     pipelnd_el1  stopped by internal counter 1/2
    # 4        1                   2     pipelnd_el1  stopped by internal counter 1/2
    # 5        2                   0     pipelnd_el2  stopped by internal counter 2/3
    # 6        0                   2     pipelnd_el1  stopped by internal counter 1/2
    # [...]

    # Check that all particles have been tracked for 4 turns
    p1.move(_context=xo.ContextCpu()) # to use numpy for the tests
    p2.move(_context=xo.ContextCpu()) # to use numpy for the tests
    p3.move(_context=xo.ContextCpu()) # to use numpy for the tests
    assert np.all(p1.at_turn == 4)
    assert np.all(p2.at_turn == 4)
    assert np.all(p3.at_turn == 4)
    assert np.all(p1.s == 8.)
    assert np.all(p2.s == 8.)
    assert np.all(p3.s == 8.)

    ############  A simpler case for some extra checks

    # Reset the counters
    line1['pipelnd_el1'].i_hold = 0
    line2['pipelnd_el2'].i_hold = 0

    p1 = xp.Particles(p0c=7e12, x=[0,0,0], _context=test_context)
    p2 = xp.Particles(p0c=7e12, x=[0,0,0], _context=test_context)

    multitracker = xt.PipelineMultiTracker(
        branches=[xt.PipelineBranch(line=line1, particles=p1),
                  xt.PipelineBranch(line=line2, particles=p2)],
        enable_debug_log=True)

    multitracker.track(num_turns=4)

    # The debug log can be loaded in a dataframe
    log_df = pd.DataFrame(multitracker.debug_log)

    # Looks like this
    #     branch  track_session_turn held_by_element                             info
    # 0        0                   0     pipelnd_el1  stopped by internal counter 1/2
    # 1        1                   0     pipelnd_el2  stopped by internal counter 1/3
    # 2        0                   1     pipelnd_el1  stopped by internal counter 1/2
    # 3        1                   0     pipelnd_el2  stopped by internal counter 2/3
    # 4        0                   2     pipelnd_el1  stopped by internal counter 1/2
    # 5        1                   1     pipelnd_el2  stopped by internal counter 1/3
    # 6        0                   3     pipelnd_el1  stopped by internal counter 1/2
    # 7        1                   1     pipelnd_el2  stopped by internal counter 2/3
    # 8        1                   2     pipelnd_el2  stopped by internal counter 1/3
    # 9        1                   2     pipelnd_el2  stopped by internal counter 2/3
    # 10       1                   3     pipelnd_el2  stopped by internal counter 1/3
    # 11       1                   3     pipelnd_el2  stopped by internal counter 2/3

    # Some checks
    assert len(multitracker.debug_log) == 12
    p1.move(_context=xo.ContextCpu()) # to use numpy for the tests
    p2.move(_context=xo.ContextCpu()) # to use numpy for the tests
    assert p1.at_turn[0] == 4
    assert p2.at_turn[0] == 4
    assert p1.s[0] == 8.
    assert p2.s[0] == 8.
    assert np.all(log_df['branch'] == np.array(4*[0, 1] + 4*[1]))
    assert np.all(log_df[log_df['branch']==0]['held_by_element'] == 'pipelnd_el1')
    assert np.all(log_df[log_df['branch']==1]['held_by_element'] == 'pipelnd_el2')
