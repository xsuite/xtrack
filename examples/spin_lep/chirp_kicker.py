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

        #include <headers/track.h>
        #include <beam_elements/elements_src/track_magnet.h>

        /*gpufun*/
        void VerticalChirpKicker_track_local_particle(
                VerticalChirpKickerData el, LocalParticle* part0){

            double const k0sl = VerticalChirpKickerData_get_k0sl(el);
            double const q_start = VerticalChirpKickerData_get_q_start(el);
            double const q_end = q_start + VerticalChirpKickerData_get_q_span(el);
            double const num_turns = VerticalChirpKickerData_get_num_turns(el);
            double const length = VerticalChirpKickerData_get_length(el);

            double dp_record = 0.;
            double dpx_record = 0.;
            double dpy_record = 0.;

            //start_per_particle_block (part0->part)
                double const at_turn = LocalParticle_get_at_turn(part);
                if (at_turn < num_turns){
                    double const old_py = LocalParticle_get_py(part);
                    // integrating to get the instantaneous phase
                    double const phi = 2 * PI * q_start * at_turn
                       + PI * (q_end - q_start) / ((double) num_turns) * ((double) at_turn * at_turn);
                    double const dpy = k0sl * sin(phi);
                    LocalParticle_add_to_py(part, dpy);

                    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
                    magnet_apply_radiation_single_particle(
                        part,
                        length,
                        0., // hx,
                        0., // hy,
                        0, // radiation_flag,
                        1, // spin_flag,
                        0., // old_px,
                        old_py,
                        0, // old_ax
                        0, // old_ay,
                        0, // old_zeta
                        0., // ks,
                        NULL, // SynchrotronRadiationRecordData record,
                        &dp_record,
                        &dpx_record,
                        &dpy_record);
                    #endif
                }
            //end_per_particle_block
        }
        ''']
