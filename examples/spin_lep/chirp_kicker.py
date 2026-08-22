import xobjects as xo
import xtrack as xt

from xtrack.random import RandomUniformAccurate, RandomExponential
from xtrack.beam_elements.elements import SynchrotronRadiationRecord

class VerticalChirpKicker(xt.BeamElement):

    _xofields = {
        'k0sl': xo.Float64,
        'q_start': xo.Float64,
        'q_span': xo.Float64,
        'num_turns': xo.Float64,
        'length': xo.Float64,
    }

    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord

    _extra_c_sources =['''

        #include "xtrack/headers/track.h"
        #include "xtrack/beam_elements/elements_src/track_magnet.h"

        GPUFUN
        void VerticalChirpKicker_track_local_particle(
                VerticalChirpKickerData el, LocalParticle* part0)
        {
            double const k0sl = VerticalChirpKickerData_get_k0sl(el);
            double const q_start = VerticalChirpKickerData_get_q_start(el);
            double const q_end = q_start + VerticalChirpKickerData_get_q_span(el);
            double const num_turns = VerticalChirpKickerData_get_num_turns(el);
            double const length = VerticalChirpKickerData_get_length(el);

            START_PER_PARTICLE_BLOCK(part0, part);
                double const at_turn = LocalParticle_get_at_turn(part);
                if (at_turn < num_turns) {
                    // integrating to get the instantaneous phase
                    double const phi = 2 * PI * q_start * at_turn
                       + PI * (q_end - q_start) / ((double) num_turns) * ((double) at_turn * at_turn);
                    double const dpy = k0sl * sin(phi);
                    LocalParticle_add_to_py(part, dpy);

                #ifndef XTRACK_MULTIPOLE_NO_SYNRAD

                    // Magnetic rigidity
                    double const p0c = LocalParticle_get_p0c(part);
                    double const q0 = LocalParticle_get_q0(part);
                    double const brho_0 = p0c / C_LIGHT / q0; // [T m]

                    // Magnetic field
                    double const Bx_T = dpy * brho_0 / length ; // [T]

                    // Track spin
                    magnet_spin(
                        part,
                        Bx_T,
                        0, // By_T
                        0, // Bz_T
                        0, // frame curvature
                        length,
                        length // lpath - same for a thin element
                    );
                #endif
                }
            END_PER_PARTICLE_BLOCK;
        }
        ''']
