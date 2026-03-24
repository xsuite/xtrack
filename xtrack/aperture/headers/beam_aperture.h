#ifndef XTRACK_BEAM_APERTURE_H
#define XTRACK_BEAM_APERTURE_H

#include <math.h>
#include <stdlib.h>

#include "xobjects/headers/common.h"
#include "base.h"
#include "path.h"
#include "polygon_algs.h"


#ifdef XO_CONTEXT_CPU_OPENMP
#define IF_OMP_PRAGMA(x) _Pragma(x)
#else
#define IF_OMP_PRAGMA(x)
#endif


typedef struct
{
    float_type x;     // closed orbit x
    float_type y;     // closed orbit y
    float_type betx;  // beta x
    float_type bety;  // beta y
    float_type dx;    // dispersion x
    float_type dy;    // dispersion y
    float_type delta; // relative energy deviation
    float_type gamma; // relativistic gamma
} TwissLocalData;


typedef struct {
    float_type emitx_norm;        // normalized emittance x
    float_type emity_norm;        // normalized emittance y
    float_type delta_rms;         // rms energy spread
    float_type tol_co;            // tolerance for closed orbit [co_radius]
    float_type tol_disp;          // tolerance for normalized dispersion [dqf]
    float_type tol_disp_ref_dx;   // tolerance for reference dispersion derivative [paras_dx]
    float_type tol_disp_ref_beta; // tolerance for reference dispersion beta [betaqfx]
    float_type tol_energy;        // tolerance for energy error [twiss_deltap]
    float_type tol_beta_beating;  // tolerance for beta beating in sigma [beta_beating]
    float_type halo_x;            // n sigma of horizontal halo
    float_type halo_y;            // n sigma of vertical halo
    float_type halo_r;            // n sigma of 45 degree halo
    float_type halo_primary;      // n sigma of primary halo
} BeamLocalData;


static inline Racetrack_s halo_racetrack(
    const TwissLocalData *twiss,
    const BeamLocalData *beam,
    const BeamApertureLocalData *aperture
)
{
    const float_type tol_dx =
        beam->tol_beta_beating *
        beam->tol_disp *
        beam->tol_disp_ref_dx *
        sqrt(twiss->betx / beam->tol_disp_ref_beta) *
        beam->delta_rms;

    const float_type tol_dy =
        beam->tol_beta_beating *
        beam->tol_disp *
        beam->tol_disp_ref_dx *
        sqrt(twiss->bety / beam->tol_disp_ref_beta) *
        beam->delta_rms;

    const float_type tol_x = aperture->tol_x;
    const float_type tol_y = aperture->tol_y;
    const float_type tol_r = aperture->tol_r;
    const float_type tol_rx = tol_r + beam->tol_co + tol_dx;
    const float_type tol_ry = tol_r + beam->tol_co + tol_dy;

    Racetrack_s rt = {
        .h = tol_x + tol_rx,
        .v = tol_y + tol_ry,
        .a = tol_rx,
        .b = tol_ry,
    };
    return rt;
}


static inline Racetrack_s beam_racetrack(
    const TwissLocalData *twiss,
    const BeamLocalData *beam
)
{
    const float_type hx = beam->halo_x / beam->halo_primary;
    const float_type hy = beam->halo_y / beam->halo_primary;
    const float_type hr = beam->halo_r / beam->halo_primary;

    const float_type ex = beam->emitx_norm / twiss->gamma;
    const float_type ey = beam->emity_norm / twiss->gamma;

    const float_type sigma_x = sqrt(ex * twiss->betx) * beam->tol_beta_beating;
    const float_type sigma_y = sqrt(ey * twiss->bety) * beam->tol_beta_beating;

    /*
        We describe the beam of the shape described by hx, hy, and hr as a
        racetrack, leading to the following equations, where sh and sv are the
        width and height of a rectangle, which, when convolved with a circle of
        radius sr (Minkowski sum) yields our racetrack:

        { hx = sh + sr  (width),
        { hy = sv + sr  (height),
        { hr = sqrt(sh ** 2 + sv ** 2) + sr  (radial maximum: through (sh, sv)).

        Solving these for sh, sv, and sr yields the following equations:
    */
    const float_type tmp = sqrt(2 * (hr - hx) * (hr - hy));
    const float_type sh  = hr - hy + tmp;
    const float_type sv  = hr - hx + tmp;
    const float_type sr  = hx + hy - hr - tmp;

    Racetrack_s rt = {
        .h = (sh + sr) * sigma_x,
        .v = (sv + sr) * sigma_y,
        .a = sr * sigma_x,
        .b = sr * sigma_y,
    };
    return rt;
}


void get_beam_envelope(
    const BeamLocalData *beam_data,
    const TwissLocalData *twiss_data,
    const BeamApertureLocalData *aperture_data,
    const float_type num_sigmas,
    int len_points,
    Point2D *out_points
)
{
    /* Beam racetrack is defined in 1-sigma units; scale by num_sigmas here */
    const Racetrack_s beam_rt_1s = beam_racetrack(twiss_data, beam_data);
    const Racetrack_s halo_rt = halo_racetrack(twiss_data, beam_data, aperture_data);

    /*
        Remembering that hx, hy, and hr are specified in sigmas, we convolve the
        beam racetrack with the aperture tolerance racetrack, to get our beam
        envelope racetrack:
    */
    const Racetrack_s env = (Racetrack_s){
        .h = halo_rt.h + num_sigmas * beam_rt_1s.h,
        .v = halo_rt.v + num_sigmas * beam_rt_1s.v,
        .a = halo_rt.a + num_sigmas * beam_rt_1s.a,
        .b = halo_rt.b + num_sigmas * beam_rt_1s.b,
    };

    Segment2D segments[8];
    Path2D path = (Path2D){
        .segments = segments,
        .len_segments = 8,
    };

    segments_from_racetrack(env.h, env.v, env.a, env.b, path.segments, &path.len_segments);
    poly_get_n_uniform_points(&path, len_points, out_points);

    /* Include the dispersive orbit shift effect */
    const Point2D disp_orbit_shift = {twiss_data->dx * beam_data->delta_rms, twiss_data->dy * beam_data->delta_rms};
    convolve_poly_and_segment(out_points, len_points, disp_orbit_shift);

    /* Shift according to the closed orbit – assuming closed orbit relative to aperture center */
    const float_type x0 = twiss_data->x;
    const float_type y0 = twiss_data->y;
    point2d_translate(x0, y0, out_points, len_points);
}


char horizontal_ray_intersects_segment(const Point2D* q, const Point2D* a, const Point2D* b)
{
    // Straddle test
    const int above_a = (a->y > q->y);
    const int above_b = (b->y > q->y);
    if (above_a == above_b) return 0;

    /* We are within the horizontal "strip" delimited by `a.y` and `b.y`.

       To check the intersection, we compare the tangent of ab segment and
       the aq segment (here assuming `b` above `a`, otherwise we need to flip
       the comparison -- done on the `return` line):

           tan_segment = (b.y - a.y) / (b.x - a.x)
           tan_point = (q.y - a.y) / (q.x - a.x)
           intersects = tan_point >= tan_segment

       To avoid division by zero we can cross-multiply:
    */
    const float_type dx = b->x - a->x;
    const float_type dy = b->y - a->y;

    const float_type lhs = dx * (q->y - a->y);
    const float_type rhs = (q->x - a->x) * dy;

    return (dy > 0) ? (lhs > rhs) : (lhs < rhs);
}


char is_point_inside_polygon(const Point2D* point, const Point2D* points, const int len_points)
/* Determine if a point is inside a polygon.

Assume the polygon is closed, i.e. that points[-1] == points[0].

Contract: len_points=len(points)
*/
{
    char inside = 0;
    for (int i = 0; i < len_points - 1; i++)
    {
        const Point2D* a = &points[i];
        const Point2D* b = &points[i + 1];
        inside ^= horizontal_ray_intersects_segment(point, a, b);
    }

    // If count is odd, point is inside (return true), otherwise return false
    return inside;
}


char _is_point_inside_polygon(const float_type* point, const float_type* points, const int len_points)
/* This function is exposed for testing purposes */
{
    return is_point_inside_polygon((const Point2D*) point, (const Point2D*) points, len_points);
}


char points_inside_polygon(const Point2D* points, const Point2D* poly_points, const int len_points, const int len_poly_points)
/* Given a set of point, determine if they are inside a polygon. False if there
is at least one point outside of the polygon, and true if all points
are contained in the polygon.

Assume the polygon is closed, i.e. that poly_points[-1] == poly_points[0].

Contract: len_points=len(points); len_poly_points=len(poly_points)
*/
{
    for (int i = 0; i < len_points; i++)
    {
        const Point2D point = points[i];
        if (!is_point_inside_polygon(&point, poly_points, len_poly_points))
            return 0;
    }
    return 1;
}


char _points_inside_polygon(const float_type* points, const float_type* poly_points, const int len_points, const int len_poly_points)
{
    return points_inside_polygon((const Point2D*) points, (const Point2D*) poly_points, len_points, len_poly_points);
}


float_type compute_max_aperture_sigma_bisection(
    const BeamLocalData *beam_data,
    const TwissLocalData *twiss_data,
    const BeamApertureLocalData *aperture_data,
    const int len_points,
    const float_type lower_bound,
    const float_type upper_bound,
    const float_type tol,
    Point2D *out_points
)
/* Obtain the maximum number of sigmas that the beam fits within the aperture.

Contract: len(out_points)=len_points; len_poly_points=len(poly_points)
*/
{
    const Point2D* poly_points = aperture_data->points;
    const float_type len_poly_points = aperture_data->n_points;

    float_type lo = lower_bound;
    float_type hi = upper_bound;

    while (hi - lo > tol) {
        const float_type mid = (lo + hi) / 2;
        get_beam_envelope(beam_data, twiss_data, aperture_data, mid, len_points, out_points);
        char inside = points_inside_polygon(out_points, poly_points, len_points, len_poly_points);

        if (inside) lo = mid;
        else hi = mid;
    }

    return lo;
}


static inline BeamLocalData beam_data_get_entry(const BeamData beam_data)
{
    return (BeamLocalData){
        .emitx_norm = BeamData_get_emitx_norm(beam_data),
        .emity_norm = BeamData_get_emity_norm(beam_data),
        .delta_rms = BeamData_get_delta_rms(beam_data),
        .tol_co = BeamData_get_tol_co(beam_data),
        .tol_disp = BeamData_get_tol_disp(beam_data),
        .tol_disp_ref_dx = BeamData_get_tol_disp_ref_dx(beam_data),
        .tol_disp_ref_beta = BeamData_get_tol_disp_ref_beta(beam_data),
        .tol_energy = BeamData_get_tol_energy(beam_data),
        .tol_beta_beating = BeamData_get_tol_beta_beating(beam_data),
        .halo_x = BeamData_get_halo_x(beam_data),
        .halo_y = BeamData_get_halo_y(beam_data),
        .halo_r = BeamData_get_halo_r(beam_data),
        .halo_primary = BeamData_get_halo_primary(beam_data),
    };
}

static inline TwissLocalData twiss_data_get_entry(const TwissData twiss_data, const uint32_t idx_slice)
{
    return (TwissLocalData){
        .x = TwissData_get_x(twiss_data, idx_slice),
        .y = TwissData_get_y(twiss_data, idx_slice),
        .betx = TwissData_get_betx(twiss_data, idx_slice),
        .bety = TwissData_get_bety(twiss_data, idx_slice),
        .dx = TwissData_get_dx(twiss_data, idx_slice),
        .dy = TwissData_get_dy(twiss_data, idx_slice),
        .delta = TwissData_get_delta(twiss_data, idx_slice),
        .gamma = TwissData_get_gamma(twiss_data),
    };
}


void compute_max_aperture_sigma(
    ApertureModel model,
    SurveyData survey,
    ProfilePolygons profile_polygons,
    ApertureBounds aperture_bounds,
    TwissData twiss_at_s,
    SurveyData survey_at_s,
    BeamData beam_data,
    float_type* const out_interpolated_apertures,
    const uint32_t envelope_num_points,
    float_type* const out_envelope_at_max_sigma,
    float_type* const sigmas
) {
    const uint32_t num_slices = TwissData_len_x(twiss_at_s);
    const uint32_t num_points = ProfilePolygons_get_num_points(profile_polygons);

    BeamLocalData s_beam_data = beam_data_get_entry(beam_data);

    #ifdef XO_CONTEXT_CPU
        int completed = 0;
    #endif

    // TODO: Make this also compatible with GPUs
    uint32_t cross_section_bound_index = 0;
    IF_OMP_PRAGMA("parallel for firstprivate(cross_section_bound_index)")
    for (uint32_t idx_slice = 0; idx_slice < num_slices; idx_slice++)
    {
        const TwissLocalData s_twiss_data = twiss_data_get_entry(twiss_at_s, idx_slice);
        BeamApertureLocalData s_aperture_data = {
            .points = (Point2D*)(out_interpolated_apertures + idx_slice * num_points * 2),
        };
        cross_section_bound_index = cross_section_at_s(
            survey_at_s,
            idx_slice,
            model,
            profile_polygons,
            aperture_bounds,
            survey,
            cross_section_bound_index,
            &s_aperture_data
        );

        const float_type num_sigmas = compute_max_aperture_sigma_bisection(
            &s_beam_data,
            &s_twiss_data,
            &s_aperture_data,
            envelope_num_points,
            /* lower bound on search */ 0,
            /* upper bound on search */ 10000,
            /* tolerance on search */ 0.001,
            (Point2D*)(out_envelope_at_max_sigma + idx_slice * envelope_num_points * 2)
        );
        sigmas[idx_slice] = num_sigmas;

        #ifdef XO_CONTEXT_CPU
            IF_OMP_PRAGMA("omp critical")
            {
                completed++;
                printf("Computing sigmas: %d%%\r", 100 * completed / num_slices);
                fflush(stdout);
            }
        #endif
    }

    #ifdef XO_CONTEXT_CPU
        printf("\n");
    #endif
}


void compute_beam_envelopes_at_sigma(
    ApertureModel model,
    ApertureBounds aperture_bounds,
    TwissData twiss_at_s,
    BeamData beam_data,
    const float_type sigmas,
    const uint32_t envelope_num_points,
    const int8_t include_aper_tols,
    float_type* const out_envelope
) {
    const uint32_t num_slices = TwissData_len_x(twiss_at_s);
    BeamLocalData s_beam_data = beam_data_get_entry(beam_data);

    #ifdef XO_CONTEXT_CPU
        int completed = 0;
    #endif

    // TODO: Make this also compatible with GPUs
    uint32_t bound_index = 0;
    IF_OMP_PRAGMA("omp parallel for firstprivate(bound_index)")
    for (uint32_t idx_slice = 0; idx_slice < num_slices; idx_slice++)
    {
        float_type s = TwissData_get_s(twiss_at_s, idx_slice);
        const TwissLocalData s_twiss_data = twiss_data_get_entry(twiss_at_s, idx_slice);

        BeamApertureLocalData s_aperture_data = {
            .points = NULL,
        };

        if (include_aper_tols) {
            bound_index = interpolate_aperture_tolerances_at_s(
                model, aperture_bounds, s, bound_index, &s_aperture_data);
        }

        Point2D* out_points = (Point2D*)(out_envelope + idx_slice * envelope_num_points * 2);
        get_beam_envelope(
            &s_beam_data,
            &s_twiss_data,
            &s_aperture_data,
            sigmas,
            envelope_num_points,
            out_points
        );

        #ifdef XO_CONTEXT_CPU
            IF_OMP_PRAGMA("omp critical")
            {
                completed++;
                printf("Computing beam envelopes: %d%%\r", 100 * completed / num_slices);
                fflush(stdout);
            }
        #endif
    }

    #ifdef XO_CONTEXT_CPU
        printf("\n");
    #endif
}


static inline float_type _envelope_at_n_error(
    float_type n,
    float_type angle,
    float_type d_target,
    Racetrack_s halo,
    Racetrack_s beam
)
/* Returns racetrack_radius_at_angle(angle, halo + n*beam) - d_target */
{
    Racetrack_s beam_at_n = scale_racetrack(beam, n);
    Racetrack_s rt = add_racetracks(halo, beam_at_n);
    return racetrack_radius_at_angle(angle, rt) - d_target;
}


static inline float_type compute_n1_for_point(
    float_type angle,
    float_type d_target,
    Racetrack_s halo,
    Racetrack_s beam,
    float_type n0,
    float_type n1
)
/* Find ``n`` such that the envelope ``halo + n * beam`` has a radius at angle
   ``angle`` equal to ``d_target``.

    Parameters:
    -----------
    angle, d_target:
        ray for which we are computing n1
    halo:
        racetrack describing the halo
    beam:
        racetrack describing the beam in 1-sigma units
    n0, n1:
        initial guesses

    Returns:
    --------
    Computed ``n1``.
 */
{
    const int n_iter = 2;
    const float_type eps = APER_PRECISION;

    float_type f0 = _envelope_at_n_error(n0, angle, d_target, halo, beam);
    float_type f1 = _envelope_at_n_error(n1, angle, d_target, halo, beam);

    for (int i = 0; i < n_iter; i++)
    {
        float_type denominator = (f1 - f0);
        if (fabs(denominator) < eps) break;

        // Newton step
        float_type n2 = n1 - f1 * (n1 - n0) / denominator;

        n0 = n1;
        f0 = f1;

        n1 = n2;
        f1 = _envelope_at_n_error(n1, angle, d_target, halo, beam);
    }

    return n1;
}


void compute_max_aperture_sigma_rays(
    ApertureModel model,
    SurveyData survey,
    ProfilePolygons profile_polygons,
    ApertureBounds aperture_bounds,
    TwissData twiss_at_s,
    SurveyData survey_at_s,
    BeamData beam_data,
    float_type* const out_interpolated_apertures,
    float_type* const ray_angles,
    const uint32_t num_ray_angles,
    float_type* const out_num_sigmas
) {
    const uint32_t num_slices = TwissData_len_x(twiss_at_s);
    const uint32_t num_points = ProfilePolygons_get_num_points(profile_polygons);

    BeamLocalData s_beam_data = beam_data_get_entry(beam_data);

    #ifdef XO_CONTEXT_CPU
        int completed = 0;
    #endif

    // TODO: Make this also compatible with GPUs
    uint32_t cross_section_bound_index = 0;
    IF_OMP_PRAGMA("omp parallel for firstprivate(cross_section_bound_index)")
    for (uint32_t idx_slice = 0; idx_slice < num_slices; idx_slice++)
    {
        const TwissLocalData s_twiss_data = twiss_data_get_entry(twiss_at_s, idx_slice);
        BeamApertureLocalData s_aperture_data = {
            .points = (Point2D*)(out_interpolated_apertures + idx_slice * num_points * 2),
        };
        cross_section_bound_index = cross_section_at_s(
            survey_at_s,
            idx_slice,
            model,
            profile_polygons,
            aperture_bounds,
            survey,
            cross_section_bound_index,
            &s_aperture_data
        );

        Racetrack_s halo_rt = halo_racetrack(&s_twiss_data, &s_beam_data, &s_aperture_data);
        Racetrack_s beam_rt = beam_racetrack(&s_twiss_data, &s_beam_data);
        Racetrack_s envelope_one_sigma_rt = add_racetracks(halo_rt, beam_rt);

        float_type aperture_distances_minus[num_ray_angles];
        float_type aperture_distances_plus[num_ray_angles];

        const float_type disp_orbit_shift_x = s_twiss_data.dx * s_beam_data.delta_rms;
        const float_type disp_orbit_shift_y = s_twiss_data.dy * s_beam_data.delta_rms;

        for (int disp_side = 0; disp_side < 2; disp_side++)
        {
            const float_type sgn = disp_side == 0 ? 1 : -1;
            dist_to_poly_along_rays(
                ray_angles,
                num_ray_angles,
                s_twiss_data.x + sgn * disp_orbit_shift_x,
                s_twiss_data.y + sgn * disp_orbit_shift_y,
                s_aperture_data.points,
                num_points,
                /* convex */ 1,
                disp_side == 0 ? aperture_distances_plus : aperture_distances_minus
            );
        }

        float_type* const slice_num_sigmas = out_num_sigmas + idx_slice * num_ray_angles;
        for (uint32_t i = 0; i < num_ray_angles; i++) {
            const float_type angle = ray_angles[i];
            const float_type d_halo = racetrack_radius_at_angle(angle, halo_rt);
            const float_type d_envelope_one_sigma = racetrack_radius_at_angle(angle, envelope_one_sigma_rt);
            const float_type d_aperture = fmin(aperture_distances_minus[i], aperture_distances_plus[i]);
            const float_type n1_lin_approx = (d_aperture - d_halo) / (d_envelope_one_sigma - d_halo);
            float_type n1 = compute_n1_for_point(angle, d_aperture, halo_rt, beam_rt, 0.f, n1_lin_approx);
            slice_num_sigmas[i] = n1;
        }

        #ifdef XO_CONTEXT_CPU
            IF_OMP_PRAGMA("omp critical")
            {
                completed++;
                printf("Computing sigmas: %d%%\r", 100 * completed / num_slices);
                fflush(stdout);
            }
        #endif
    }

    #ifdef XO_CONTEXT_CPU
        printf("\n");
    #endif
}


void compute_max_aperture_sigma_exact(
    ApertureModel model,
    SurveyData survey,
    ProfilePolygons profile_polygons,
    ApertureBounds aperture_bounds,
    TwissData twiss_at_s,
    SurveyData survey_at_s,
    BeamData beam_data,
    float_type* const out_interpolated_apertures,
    float_type* const ray_angles,
    const uint32_t num_ray_angles,
    float_type* const out_num_sigmas
) {
    const uint32_t num_slices = TwissData_len_x(twiss_at_s);
    const uint32_t num_points = ProfilePolygons_get_num_points(profile_polygons);

    BeamLocalData s_beam_data = beam_data_get_entry(beam_data);

    #ifdef XO_CONTEXT_CPU
        int completed = 0;
    #endif

    uint32_t cross_section_bound_index = 0;
    IF_OMP_PRAGMA("omp parallel for firstprivate(cross_section_bound_index)")
    for (uint32_t idx_slice = 0; idx_slice < num_slices; idx_slice++)
    {
        const TwissLocalData s_twiss_data = twiss_data_get_entry(twiss_at_s, idx_slice);
        BeamApertureLocalData s_aperture_data = {
            .points = (Point2D*)(out_interpolated_apertures + idx_slice * num_points * 2),
        };
        cross_section_bound_index = cross_section_at_s(
            survey_at_s,
            idx_slice,
            model,
            profile_polygons,
            aperture_bounds,
            survey,
            cross_section_bound_index,
            &s_aperture_data
        );

        const float_type ex = s_beam_data.emitx_norm / s_twiss_data.gamma;
        const float_type ey = s_beam_data.emity_norm / s_twiss_data.gamma;
        const float_type sigma_x = sqrt(ex * s_twiss_data.betx) * s_beam_data.tol_beta_beating;
        const float_type sigma_y = sqrt(ey * s_twiss_data.bety) * s_beam_data.tol_beta_beating;

        const float_type disp_orbit_shift_x = s_twiss_data.dx * s_beam_data.delta_rms;
        const float_type disp_orbit_shift_y = s_twiss_data.dy * s_beam_data.delta_rms;

        const Racetrack_s halo_rt = halo_racetrack(&s_twiss_data, &s_beam_data, &s_aperture_data);
        const Racetrack_s beam_1s_rt = beam_racetrack(&s_twiss_data, &s_beam_data);

        float_type n1 = INFINITY;
        for (uint32_t i = 0; i < num_ray_angles; i++)
        {
            const float_type angle0 = ray_angles[i];
            const float_type d_halo = racetrack_radius_at_angle(angle0, halo_rt);
            const float_type cos0 = cos(angle0);
            const float_type sin0 = sin(angle0);

            for (uint32_t j = 0; j < num_ray_angles; j++)
            {
                const float_type angle1 = ray_angles[j];
                const float_type p_minus_x = s_twiss_data.x - disp_orbit_shift_x + d_halo * cos0;
                const float_type p_minus_y = s_twiss_data.y - disp_orbit_shift_y + d_halo * sin0;
                const float_type p_plus_x = s_twiss_data.x + disp_orbit_shift_x + d_halo * cos0;
                const float_type p_plus_y = s_twiss_data.y + disp_orbit_shift_y + d_halo * sin0;

                const RayHit_s hit_minus = dist_to_poly_along_ray(
                    angle1,
                    p_minus_x,
                    p_minus_y,
                    s_aperture_data.points,
                    s_aperture_data.n_points,
                    /* convex */ 1,
                    /* start_at */ 0
                );
                const RayHit_s hit_plus = dist_to_poly_along_ray(
                    angle1,
                    p_plus_x,
                    p_plus_y,
                    s_aperture_data.points,
                    s_aperture_data.n_points,
                    /* convex */ 1,
                    /* start_at */ 0
                );
                const float_type d_aperture = fmin(hit_minus.dist, hit_plus.dist);
                const float_type sigma_ray = racetrack_radius_at_angle(angle1, beam_1s_rt);

                const float_type n_ray = d_aperture / sigma_ray;
                if (n_ray < n1)
                    n1 = n_ray;
            }
        }

        out_num_sigmas[idx_slice] = n1;

        #ifdef XO_CONTEXT_CPU
            IF_OMP_PRAGMA("omp critical")
            {
                completed++;
                printf("Computing sigmas: %d%%\r", 100 * completed / num_slices);
                fflush(stdout);
            }
        #endif
    }

    #ifdef XO_CONTEXT_CPU
        printf("\n");
    #endif
}

#endif /* XTRACK_BEAM_APERTURE_H */
