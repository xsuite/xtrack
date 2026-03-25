import xobjects as xo
from xtrack.aperture.structures import (
    ApertureBounds,
    ApertureModel,
    BeamData,
    FloatType,
    Profile,
    ProfilePolygons,
    SurveyData,
    TwissData
)


def build_aperture_kernels(context):
    source = '''
        #include "xtrack/aperture/headers/profiles.h"
        #include "xtrack/aperture/headers/beam_aperture.h"
        #include "xtrack/aperture/headers/survey_tools.h"
    '''

    kernels = {
        "build_profile_polygons": xo.Kernel(
            c_name="build_profile_polygons",
            args=[
                xo.Arg(ApertureModel, name="model"),
                xo.Arg(ProfilePolygons, name="profile_polygons"),
                xo.Arg(ApertureBounds, name="aperture_bounds"),
                xo.Arg(SurveyData, name="survey"),
            ],
        ),
        "cross_sections_at_s": xo.Kernel(
            c_name="cross_sections_at_s",
            args=[
                xo.Arg(SurveyData, name="survey_at_s"),
                xo.Arg(ApertureModel, name="model"),
                xo.Arg(ProfilePolygons, name="profile_polygons"),
                xo.Arg(ApertureBounds, name="aperture_bounds"),
                xo.Arg(SurveyData, name="survey"),
                xo.Arg(FloatType, pointer=True, name="cross_sections"),
                xo.Arg(FloatType, pointer=True, name="tol_r"),
                xo.Arg(FloatType, pointer=True, name="tol_x"),
                xo.Arg(FloatType, pointer=True, name="tol_y"),
            ],
        ),
        "compute_max_aperture_sigma_bisection": xo.Kernel(
            c_name="compute_max_aperture_sigma_bisection",
            args=[
                xo.Arg(ApertureModel, name="model"),
                xo.Arg(SurveyData, name="survey"),
                xo.Arg(ProfilePolygons, name="profile_polygons"),
                xo.Arg(ApertureBounds, name="aperture_bounds"),
                xo.Arg(TwissData, name="twiss_at_s"),
                xo.Arg(SurveyData, name="survey_at_s"),
                xo.Arg(BeamData, name="beam_data"),
                xo.Arg(FloatType, pointer=True, name="out_interpolated_apertures"),
                xo.Arg(xo.UInt32, name="envelope_num_points"),
                xo.Arg(FloatType, pointer=True, name="out_envelope_at_max_sigma"),
                xo.Arg(FloatType, pointer=True, name="sigmas"),
            ],
        ),
        "compute_max_aperture_sigma_rays": xo.Kernel(
            c_name="compute_max_aperture_sigma_rays",
            args=[
                xo.Arg(ApertureModel, name="model"),
                xo.Arg(SurveyData, name="survey"),
                xo.Arg(ProfilePolygons, name="profile_polygons"),
                xo.Arg(ApertureBounds, name="aperture_bounds"),
                xo.Arg(TwissData, name="twiss_at_s"),
                xo.Arg(SurveyData, name="survey_at_s"),
                xo.Arg(BeamData, name="beam_data"),
                xo.Arg(FloatType, pointer=True, name="out_interpolated_apertures"),
                xo.Arg(xo.UInt32, name="envelope_num_points"),
                xo.Arg(FloatType, pointer=True, name="out_envelope_at_max_sigma"),
                xo.Arg(FloatType, pointer=True, name="ray_angles"),
                xo.Arg(xo.UInt32, name="num_ray_angles"),
                xo.Arg(FloatType, pointer=True, name="sigmas"),
            ],
        ),
        "compute_max_aperture_sigma_exact": xo.Kernel(
            c_name="compute_max_aperture_sigma_exact",
            args=[
                xo.Arg(ApertureModel, name="model"),
                xo.Arg(SurveyData, name="survey"),
                xo.Arg(ProfilePolygons, name="profile_polygons"),
                xo.Arg(ApertureBounds, name="aperture_bounds"),
                xo.Arg(TwissData, name="twiss_at_s"),
                xo.Arg(SurveyData, name="survey_at_s"),
                xo.Arg(BeamData, name="beam_data"),
                xo.Arg(FloatType, pointer=True, name="out_interpolated_apertures"),
                xo.Arg(xo.UInt32, name="envelope_num_points"),
                xo.Arg(FloatType, pointer=True, name="out_envelope_at_max_sigma"),
                xo.Arg(FloatType, pointer=True, name="ray_angles"),
                xo.Arg(xo.UInt32, name="num_ray_angles"),
                xo.Arg(FloatType, pointer=True, name="sigmas"),
            ],
        ),
        "compute_beam_envelopes_at_sigma": xo.Kernel(
            c_name="compute_beam_envelopes_at_sigma",
            args=[
                xo.Arg(ApertureModel, name='model'),
                xo.Arg(ApertureBounds, name='aperture_bounds'),
                xo.Arg(TwissData, name='twiss_at_s'),
                xo.Arg(BeamData, name='beam_data'),
                xo.Arg(FloatType, name='sigmas'),
                xo.Arg(xo.UInt32, name='envelope_num_points'),
                xo.Arg(xo.Int8, name='include_aper_tols'),
                xo.Arg(FloatType, pointer=True, name='out_envelope'),
            ]
        ),
        "build_polygon_for_profile": xo.Kernel(
            c_name="build_polygon_for_profile",
            args=[
                xo.Arg(FloatType, pointer=True, name="points"),
                xo.Arg(xo.UInt32, name="len_points"),
                xo.Arg(Profile, name="profile"),
            ],
        ),
        "_points_inside_polygon": xo.Kernel(
            c_name="_points_inside_polygon",
            args=[
                xo.Arg(FloatType, pointer=True, name="points"),
                xo.Arg(FloatType, pointer=True, name="poly_points"),
                xo.Arg(xo.UInt32, name="len_points"),
                xo.Arg(xo.UInt32, name="len_poly_points"),
            ],
            ret=xo.Arg(xo.Int8),
        ),
        "_is_point_inside_polygon": xo.Kernel(
            c_name="_is_point_inside_polygon",
            args=[
                xo.Arg(FloatType, pointer=True, name="point"),
                xo.Arg(FloatType, pointer=True, name="points"),
                xo.Arg(xo.UInt32, name="len_points"),
            ],
            ret=xo.Arg(xo.Int8),
        ),
        "resample_survey_table": xo.Kernel(
            c_name="resample_survey_table",
            args=[
                xo.Arg(SurveyData, name="survey"),
                xo.Arg(FloatType, pointer=True, name="s"),
                xo.Arg(SurveyData, name="sliced"),
            ]
        ),
    }

    context.add_kernels(
        sources=[source],
        kernels=kernels,
    )
