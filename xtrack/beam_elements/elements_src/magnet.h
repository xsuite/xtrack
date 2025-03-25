// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_MAGNET_H
#define XTRACK_MAGNET_H

/*gpufun*/
void Magnet_track_local_particle(
    MagnetData el,
    LocalParticle* part0
) {
    const double length = MagnetData_get_length(el);
    const int64_t order = MagnetData_get_order(el);
    const double inv_factorial_order = MagnetData_get_inv_factorial_order(el);
    /*gpuglmem*/ const double* knl = MagnetData_getp1_knl(el, 0);
    /*gpuglmem*/ const double* ksl = MagnetData_getp1_ksl(el, 0);
    const int64_t num_multipole_kicks = MagnetData_get_num_multipole_kicks(el);
    const double h = MagnetData_get_h(el);
    const double k0 = MagnetData_get_k0(el);
    const double k1 = MagnetData_get_k1(el);
    const double k2 = MagnetData_get_k2(el);
    const double k3 = MagnetData_get_k3(el);
    const double k0s = MagnetData_get_k0s(el);
    const double k1s = MagnetData_get_k1s(el);
    const double k2s = MagnetData_get_k2s(el);
    const double k3s = MagnetData_get_k3s(el);
    const int64_t model = MagnetData_get_model(el);
    const int64_t integrator = MagnetData_get_integrator(el);
    const int64_t radiation_flag = MagnetData_get_radiation_flag(el);
    const double delta_taper = MagnetData_get_delta_taper(el);

    int64_t edge_entry_active = MagnetData_get_edge_entry_active(el);
    int64_t edge_exit_active = MagnetData_get_edge_exit_active(el);
    int64_t edge_entry_model = MagnetData_get_edge_entry_model(el);
    int64_t edge_exit_model = MagnetData_get_edge_exit_model(el);
    double edge_entry_angle = MagnetData_get_edge_entry_angle(el);
    double edge_exit_angle = MagnetData_get_edge_exit_angle(el);
    double edge_entry_angle_fdown = MagnetData_get_edge_entry_angle_fdown(el);
    double edge_exit_angle_fdown = MagnetData_get_edge_exit_angle_fdown(el);
    double edge_entry_fint = MagnetData_get_edge_entry_fint(el);
    double edge_exit_fint = MagnetData_get_edge_exit_fint(el);
    double edge_entry_hgap = MagnetData_get_edge_entry_hgap(el);
    double edge_exit_hgap = MagnetData_get_edge_exit_hgap(el);

    track_magnet_particles(
        part0,
        length,
        order,
        inv_factorial_order,
        knl,
        ksl,
        /*factor_knl_ksl*/ 1.,
        num_multipole_kicks,
        model,
        integrator,
        radiation_flag,
        NULL, // radiation_record
        delta_taper,
        h,
        0., // hxl
        k0,
        k1,
        k2,
        k3,
        k0s,
        k1s,
        k2s,
        k3s,
        edge_entry_active,
        edge_exit_active,
        edge_entry_model,
        edge_exit_model,
        edge_entry_angle,
        edge_exit_angle,
        edge_entry_angle_fdown,
        edge_exit_angle_fdown,
        edge_entry_fint,
        edge_exit_fint,
        edge_entry_hgap,
        edge_exit_hgap
    );
}

#endif