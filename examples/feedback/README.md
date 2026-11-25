<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Feedback

Start by binding a BPM and a corrector, then measure and use the orbit response.
Continue with two-BPM correction, actuator dynamics and a weak response
mode. The later lessons cover bunch and slice resolution, several feedbacks
on one line, strong–strong collisions and acceleration. Each script runs
independently.

| Example | Lesson | Approximate CPU time |
| --- | --- | --- |
| [001_feedback.py](001_feedback.py) | Bind a BPM and corrector and track across the first update | A few seconds |
| [002_correct_two_bpms.py](002_correct_two_bpms.py) | Measure the signed response matrix and correct two BPMs on the demo ring | A few seconds |
| [003_orbit_feedback.py](003_orbit_feedback.py) | Follow a PI correction from measurement through delays and slew to the applied field | About 30 s |
| [004_closed_orbit_feedback.py](004_closed_orbit_feedback.py) | Use truncated SVD when two correctors have nearly the same response | About 1 min |
| [005_tune_feedback.py](005_tune_feedback.py) | Estimate and correct two tunes using a known Exciter waveform and a noisy BPM | About 40 s |
| [006_bunch_by_bunch_orbit_feedback.py](006_bunch_by_bunch_orbit_feedback.py) | Separate common orbit feedback from individual bunch correction on a sparse fill | About 8 s |
| [007_collision_timing_feedback.py](007_collision_timing_feedback.py) | Compare RF and chicane trims, then align the two arrivals with a master clock | About 3 min |
| [008_head_tail_feedback.py](008_head_tail_feedback.py) | Use a sliced pickup and an odd kick profile to damp intrabunch motion | A few seconds |
| [009_combined_feedbacks.py](009_combined_feedbacks.py) | Run orbit, tune and head–tail feedback on the same bunch | About 15 s |
| [010_ip_feedback.py](010_ip_feedback.py) | Control vertical IP separation with ideal and quantized BPMs during strong–strong collisions | Several minutes on CPU |
| [011_feedback_on_energy_ramp.py](011_feedback_on_energy_ramp.py) | Compare damping and excitation as the revolution period and command delivery lag change | A few seconds |

Run a file from the checkout root with Python or IPython in the environment
where Xsuite and this Xtrack checkout are available. The figures use Matplotlib.
Example 010 also requires Xfields.

```bash
ipython examples/feedback/001_feedback.py
```

In an IPython session, `%run examples/feedback/001_feedback.py` leaves the
line, particles, feedback and figure available for inspection. Edit the
**User parameters** near the top of each file to vary the comparison.

Example 002 loads the [demo ring](../../test_data/example_lattices/demo_ring/README.md),
a 1 GeV/c electron ring with RF voltage and radiation off. The orbit feedbacks
use turn averages to distinguish the slow orbit from coherent betatron motion.
The other examples build small models inline to isolate the effects being studied.

Examples 001–006, 008, 009 and 011 place `xt.feedback.Feedback` on the line
and use ordinary `line.track(...)`. Example 005 uses a Python controller
and an acquisition transform to pair the BPM and drive samples. The
RF/chicane and collision studies use the experimental shared boundary.

In 002, one controller solves for the two vertical corrector kicks using both BPM
readings. In 009, three controllers use a shared pickup with different
observables, calculation intervals and actuators.

Example 010 compares feedback off, ideal position BPMs and 200 nm BPM
increments. It uses four OpenMP threads by default. Set `OMP_THREADS = 0`
for a serial context or `OMP_THREADS = 'auto'` to use the available threads.
Set `STUDY = 'shared-correctors'` to see the single-turn correction limit
with two bunches in each ring and one corrector setting at each of four IPs.

Example 011 tracks the same launch on one tracker at two vertical tunes,
with feedback off, immediate delivery and a fixed 0.52 µs command delay. The reference momentum follows
an `EnergyProgram` and the corrector uses the rigidity at each passage.
`line.feedback.reset()` clears the pickups and feedback state between runs.
At a vertical tune of 0.14, the delayed loop damps the oscillation through
the change in command lag. At 0.25, the same gain and delay drive the
oscillation after the additional turn of latency.
