// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_RBEND_H
#define XTRACK_RBEND_H

/*gpufun*/
void RBend_track_local_particle(
        RBendData el,
        LocalParticle* part0
) {

    const double k0 = RBendData_get_k0(el);
    double length = RBendData_get_length(el);
    const double h = RBendData_get_h(el);
    const double angle = length * h;

    // Edge at entry
    #ifdef XSUITE_BACKTRACK
        const int64_t edge_entry_active = RBendData_get_edge_exit_active(el);
    #else
        const int64_t edge_entry_active = RBendData_get_edge_entry_active(el);
    #endif
    if (edge_entry_active){
        #ifdef XSUITE_BACKTRACK
            int64_t const edge_entry_model = RBendData_get_edge_exit_model(el);
            double const edge_entry_angle = RBendData_get_edge_exit_angle(el) + angle / 2;
            double const edge_entry_angle_fdown = RBendData_get_edge_exit_angle_fdown(el);
            double const edge_entry_fint = RBendData_get_edge_exit_fint(el);
            double const edge_entry_hgap = RBendData_get_edge_exit_hgap(el);
        #else
            int64_t const edge_entry_model = RBendData_get_edge_entry_model(el);
            double const edge_entry_angle = RBendData_get_edge_entry_angle(el) + angle / 2;
            double const edge_entry_angle_fdown = RBendData_get_edge_entry_angle_fdown(el);
            double const edge_entry_fint = RBendData_get_edge_entry_fint(el);
            double const edge_entry_hgap = RBendData_get_edge_entry_hgap(el);
        #endif

        if (edge_entry_model==0){
            double r21, r43;
            compute_dipole_edge_linear_coefficients(k0, edge_entry_angle,
                    edge_entry_angle_fdown, edge_entry_hgap, edge_entry_fint,
                    &r21, &r43);
            #ifdef XSUITE_BACKTRACK
                r21 = -r21;
                r43 = -r43;
            #endif
            //start_per_particle_block (part0->part)
                DipoleEdgeLinear_single_particle(part, r21, r43);
            //end_per_particle_block
        }
        else if (edge_entry_model==1){
            #ifdef XSUITE_BACKTRACK
                //start_per_particle_block (part0->part)
                    LocalParticle_kill_particle(part, -32);
                //end_per_particle_block
                return;
            #else
                //start_per_particle_block (part0->part)
                    DipoleEdgeNonLinear_single_particle(part, k0, edge_entry_angle,
                                        edge_entry_fint, edge_entry_hgap, 0);
                //end_per_particle_block
            #endif
        }
    } // end edge entry

    double factor_knl_ksl = 1;

    #ifdef XSUITE_BACKTRACK
        length = -length;
        factor_knl_ksl = -1;
    #endif

    const double k1 = RBendData_get_k1(el);

    int64_t num_multipole_kicks = RBendData_get_num_multipole_kicks(el);
    const int64_t order = RBendData_get_order(el);
    const double inv_factorial_order = RBendData_get_inv_factorial_order(el);

    const int64_t model = RBendData_get_model(el);

    /*gpuglmem*/ const double *knl = RBendData_getp1_knl(el, 0);
    /*gpuglmem*/ const double *ksl = RBendData_getp1_ksl(el, 0);

    Bend_track_local_particle_from_params(part0,
                                    length, k0, k1, h,
                                    num_multipole_kicks, model,
                                    knl, ksl,
                                    order, inv_factorial_order,
                                    factor_knl_ksl);

    // Edge at exit
    #ifdef XSUITE_BACKTRACK
        const int64_t edge_exit_active = RBendData_get_edge_entry_active(el);
    #else
        const int64_t edge_exit_active = RBendData_get_edge_exit_active(el);
    #endif
    if (edge_exit_active){
        #ifdef XSUITE_BACKTRACK
            int64_t const edge_exit_model = RBendData_get_edge_entry_model(el);
            double const edge_exit_angle = RBendData_get_edge_entry_angle(el) + angle / 2;
            double const edge_exit_angle_fdown = RBendData_get_edge_entry_angle_fdown(el);
            double const edge_exit_fint = RBendData_get_edge_entry_fint(el);
            double const edge_exit_hgap = RBendData_get_edge_entry_hgap(el);
        #else
            int64_t const edge_exit_model = RBendData_get_edge_exit_model(el);
            double const edge_exit_angle = RBendData_get_edge_exit_angle(el) + angle / 2;
            double const edge_exit_angle_fdown = RBendData_get_edge_exit_angle_fdown(el);
            double const edge_exit_fint = RBendData_get_edge_exit_fint(el);
            double const edge_exit_hgap = RBendData_get_edge_exit_hgap(el);
        #endif

        if (edge_exit_model==0){
            double r21, r43;
            compute_dipole_edge_linear_coefficients(k0, edge_exit_angle,
                    edge_exit_angle_fdown, edge_exit_hgap, edge_exit_fint,
                    &r21, &r43);
            #ifdef XSUITE_BACKTRACK
                r21 = -r21;
                r43 = -r43;
            #endif
            //start_per_particle_block (part0->part)
                DipoleEdgeLinear_single_particle(part, r21, r43);
            //end_per_particle_block
        }
        else if (edge_exit_model==1){
            #ifdef XSUITE_BACKTRACK
                //start_per_particle_block (part0->part)
                    LocalParticle_kill_particle(part, -32);
                //end_per_particle_block
                return;
            #else
                //start_per_particle_block (part0->part)
                    DipoleEdgeNonLinear_single_particle(part, k0, edge_exit_angle,
                                        edge_exit_fint, edge_exit_hgap, 1);
                //end_per_particle_block
            #endif
        }
    } // end edge exit

}

#endif // XTRACK_RBEND_H
