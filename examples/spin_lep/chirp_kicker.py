import xobjects as xo
import xtrack as xt

class VerticalChirpKicker(xt.BeamElement):

    _xofields = {
        'k0sl': xo.Float64,
        'q_start': xo.Float64,
        'q_span': xo.Float64,
        'num_turns': xo.Float64,
    }

    _extra_c_sources =['''
        /*gpufun*/
        void VerticalChirpKicker_track_local_particle(
                VerticalChirpKickerData el, LocalParticle* part0){

            double const k0sl = VerticalChirpKickerData_get_k0sl(el);
            double const q_start = VerticalChirpKickerData_get_q_start(el);
            double const q_end = q_start + VerticalChirpKickerData_get_q_span(el);
            double const num_turns = VerticalChirpKickerData_get_num_turns(el);

            //start_per_particle_block (part0->part)
                double const at_turn = LocalParticle_get_at_turn(part);
                if (at_turn < num_turns){
                    double const qq = q_start + (q_end - q_start) * ((double) at_turn) / ((double) num_turns);
                    double const dpy = k0sl * sin(2 * PI * qq * at_turn);
                    LocalParticle_add_to_py(part, dpy);
                }
            //end_per_particle_block
        }
        ''']
