import xobjects as xo
from xtrack.aperture.structures import (
    ApertureModel,
    BeamData,
    CrossSections,
    Profile,
    SurveyData,
    TwissData
)


def build_aperture_kernels(context):
    source = '''
        #include "xtrack/aperture/headers/polygons.h"
        #include "xtrack/aperture/headers/beam_aperture.h"
        #include "xtrack/aperture/headers/survey_tools.h"
    '''

    kernels = {
        "build_profile_polygons": xo.Kernel(
            c_name="build_profile_polygons",
            args=[
                xo.Arg(ApertureModel, name="model"),
                xo.Arg(CrossSections, name="cross_sections"),
            ],
        ),
        "compute_max_aperture_sigma": xo.Kernel(
            c_name="compute_max_aperture_sigma",
            args=[
                xo.Arg(ApertureModel, name="model"),
                xo.Arg(CrossSections, name="cross_sections"),
                xo.Arg(TwissData, name="twiss_data"),
                xo.Arg(BeamData, name="beam_data"),
                xo.Arg(xo.Float32, pointer=True, name="out_interpolated_apertures"),
                xo.Arg(xo.UInt32, name="envelope_num_points"),
                xo.Arg(xo.Float32, pointer=True, name="out_envelope_at_max_sigma"),
                xo.Arg(xo.Float32, pointer=True, name="sigmas"),
            ],
        ),
        "compute_horizontal_vertical_diagonal_aperture_sigmas": xo.Kernel(
            c_name="compute_horizontal_vertical_diagonal_aperture_sigmas",
            args=[
                xo.Arg(ApertureModel, name="model"),
                xo.Arg(CrossSections, name="cross_sections"),
                xo.Arg(TwissData, name="twiss_data"),
                xo.Arg(BeamData, name="beam_data"),
                xo.Arg(xo.Float32, pointer=True, name="out_interpolated_apertures"),
                xo.Arg(xo.Float32, pointer=True, name="out_sigmas_h"),
                xo.Arg(xo.Float32, pointer=True, name="out_sigmas_v"),
                xo.Arg(xo.Float32, pointer=True, name="out_sigmas_d"),
            ],
        ),
        "compute_beam_envelopes_at_sigma": xo.Kernel(
            c_name="compute_beam_envelopes_at_sigma",
            args=[
                xo.Arg(ApertureModel, name='model'),
                xo.Arg(CrossSections, name='cross_sections'),
                xo.Arg(TwissData, name='twiss_data'),
                xo.Arg(BeamData, name='beam_data'),
                xo.Arg(xo.Float32, name='sigmas'),
                xo.Arg(xo.Float32, pointer=True, name='out_interpolated_apertures'),
                xo.Arg(xo.UInt32, name='envelope_num_points'),
                xo.Arg(xo.Float32, pointer=True, name='out_envelope'),
            ]
        ),
        "build_polygon_for_profile": xo.Kernel(
            c_name="build_polygon_for_profile",
            args=[
                xo.Arg(xo.Float32, pointer=True, name="points"),
                xo.Arg(xo.UInt32, name="num_points"),
                xo.Arg(Profile, name="profile"),
            ],
        ),
        "_points_inside_polygon": xo.Kernel(
            c_name="_points_inside_polygon",
            args=[
                xo.Arg(xo.Float32, pointer=True, name="points"),
                xo.Arg(xo.Float32, pointer=True, name="poly_points"),
                xo.Arg(xo.UInt32, name="len_points"),
                xo.Arg(xo.UInt32, name="len_poly_points"),
            ],
            ret=xo.Arg(xo.Int8),
        ),
        "_is_point_inside_polygon": xo.Kernel(
            c_name="_is_point_inside_polygon",
            args=[
                xo.Arg(xo.Float32, pointer=True, name="point"),
                xo.Arg(xo.Float32, pointer=True, name="points"),
                xo.Arg(xo.UInt32, name="len_points"),
            ],
            ret=xo.Arg(xo.Int8),
        ),
        "resample_survey_table": xo.Kernel(
            c_name="resample_survey_table",
            args=[
                xo.Arg(SurveyData, name="survey"),
                xo.Arg(xo.Float32, pointer=True, name="s"),
                xo.Arg(SurveyData, name="sliced"),
            ]
        ),
    }

    context.add_kernels(
        sources=[source],
        kernels=kernels,
    )
