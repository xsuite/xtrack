#ifndef SURVEY_TOOLS_H
#define SURVEY_TOOLS_H

#include "base.h"
#include "segment3d.h"


typedef struct {
    float_type s;                   // position along the beamline}
    float_type angle;               // angle of the pose vector
    float_type length;              // length of the segment
    float_type tilt;                // tilt of the segment
    Pose pose;                      // pose matrix (4x4) for each entry
    // The following are only used if the survey segment represents an straight-body RBend (or a similar object, which
    // curves the reference frame externally, but has a straight body internally). If unused, they are set to NAN.
    float_type rbend_shift_x_in;    // for an RBend: x-shift between the pose origin and the magnet axis
    float_type rbend_angle_in;      // for an RBend: angle between the survey so far (s at pose) and the magnet axis
} SurveyEntry_s;


static inline Pose survey_arc_matrix(
    const float_type length,
    const float_type angle,
    const float_type tilt
) {
    if (fabs(angle) < APER_PRECISION) {
        return transform_to_matrix((Transform) {.s = length});
    }

    const float_type ct = cos(tilt), st = sin(tilt);
    const float_type ca = cos(angle), sa = sin(angle);
    const float_type dx = length * (ca - 1) / angle;
    const float_type ds = length * sa / angle;
    return (Pose) {
        .mat = {
            {
                ct * ct * ca + st * st,
                ct * st * (ca - 1),
                -ct * sa,
                ct * dx,
            },
            {
                ct * st * (ca - 1),
                st * st * ca + ct * ct,
                -st * sa,
                st * dx,
            },
            {ct * sa, st * sa, ca, ds},
            {0, 0, 0, 1},
        }
    };
}


static inline float_type get_survey_max_s(const SurveyData survey)
/* Get the max s of the survey. */
{
    // Rely on the fact that the last element is `_end_point` (length = 0).
    // This needs to change, if that is ever not the case.
    const uint32_t survey_num_entries = SurveyData_len_s(survey);
    return SurveyData_get_s(survey, survey_num_entries - 1);
}


static inline Pose pose_matrix_from_survey(const SurveyData survey, const uint32_t idx) {
    Pose m;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            m.mat[i][j] = SurveyData_get_pose(survey, idx, i, j);
        }
    }
    return m;
}


static inline LineSegment3D survey_line_segment(SurveyData survey, uint32_t idx) {
    const Pose start_pose = pose_matrix_from_survey(survey, idx);
    const float_type length = SurveyData_get_length(survey, idx);
    const Point3D entry = pose_apply_point(start_pose, (Point3D) {0, 0, 0});
    const Point3D exit = pose_apply_point(start_pose, (Point3D) {0, 0, length});
    return (LineSegment3D) {
        .start = entry,
        .end = exit
    };
}


static inline LineSegment3D survey_straight_rbend_segment(
    SurveyData survey,
    uint32_t idx
) {
    const float_type angle = SurveyData_get_angle(survey, idx);
    const float_type length = SurveyData_get_length(survey, idx);
    const float_type angle_in =
        SurveyData_get_rbend_angle_in(survey, idx);
    const float_type angle_out = angle - angle_in;
    const float_type length_straight =
        length / angle * (sin(angle_in) + sin(angle_out));
    const float_type tilt = SurveyData_get_tilt(survey, idx);
    const float_type shift_x_in =
        SurveyData_get_rbend_shift_x_in(survey, idx);

    Pose body_start_pose = matrix_multiply(
        pose_matrix_from_survey(survey, idx),
        survey_arc_matrix(0, angle_in, tilt));
    body_start_pose = matrix_multiply(
        body_start_pose,
        transform_to_matrix((Transform) {
            .x = -shift_x_in * cos(tilt),
            .y = -shift_x_in * sin(tilt),
        }));

    const Point3D body_start = pose_apply_point(
        body_start_pose, (Point3D) {0, 0, 0});
    const Point3D body_end = pose_apply_point(
        body_start_pose, (Point3D) {0, 0, length_straight});
    return (LineSegment3D) {
        .start = body_start,
        .end = body_end
    };
}


static inline ArcSegment3D survey_arc_segment(SurveyData survey, uint32_t idx) {
    const Pose start = pose_matrix_from_survey(survey, idx);
    const float_type length = SurveyData_get_length(survey, idx);
    const float_type angle = SurveyData_get_angle(survey, idx);
    const float_type roll = SurveyData_get_tilt(survey, idx);

    return (ArcSegment3D) {
        .start = start,
        .length = length,
        .curvature = angle / length,
        .roll = roll
    };
}


static inline Segment3D survey_segment(SurveyData survey, uint32_t idx) {
    const float_type angle = SurveyData_get_angle(survey, idx);
    const float_type length = SurveyData_get_length(survey, idx);
    const float_type rbend_angle_in = SurveyData_get_rbend_angle_in(survey, idx);

    // The stored poses include the RBend edge rotations; construct the
    // internal straight body from its entrance alignment instead.
    if (isfinite(rbend_angle_in) && fabs(angle) >= APER_PRECISION)
    {
        return (Segment3D) {
            .type = SEGMENT3D_LINE,
            .line = survey_straight_rbend_segment(survey, idx)
        };
    }
    else if (fabs(angle) < APER_PRECISION
        || fabs(length) < APER_PRECISION)
    {
        return (Segment3D) {
            .type = SEGMENT3D_LINE,
            .line = survey_line_segment(survey, idx)
        };
    }
    else
    {
        return (Segment3D) {
            .type = SEGMENT3D_ARC,
            .arc = survey_arc_segment(survey, idx)
        };
    }
}


SurveyEntry_s interpolate_survey_table_entry(
    const SurveyData survey,
    const float_type s_target,
    const uint32_t i_survey
)
{
    const float_type eps = APER_PRECISION;
    const float_type s_current = SurveyData_get_s(survey, i_survey);
    const float_type s_next = SurveyData_get_s(survey, i_survey + 1);

    SurveyEntry_s entry;
    entry.tilt = SurveyData_get_tilt(survey, i_survey);
    entry.rbend_shift_x_in = SurveyData_get_rbend_shift_x_in(survey, i_survey);
    entry.rbend_angle_in = SurveyData_get_rbend_angle_in(survey, i_survey);

    if (fabs(s_target - s_current) < eps) {
        // Target is current, simply copy the current survey entry
        entry.s = SurveyData_get_s(survey, i_survey);
        entry.angle = SurveyData_get_angle(survey, i_survey);
        entry.length = SurveyData_get_length(survey, i_survey);
        entry.pose = pose_matrix_from_survey(survey, i_survey);
        // tilt and RBend related fields are already in
    }
    else if (fabs(s_target - s_next) < eps) {
        // Target is next, simply copy the next survey entry
        entry.s = SurveyData_get_s(survey, i_survey + 1);
        entry.angle = SurveyData_get_angle(survey, i_survey + 1);
        entry.length = SurveyData_get_length(survey, i_survey + 1);
        entry.tilt = SurveyData_get_tilt(survey, i_survey + 1);
        entry.rbend_shift_x_in = SurveyData_get_rbend_shift_x_in(survey, i_survey + 1);
        entry.rbend_angle_in = SurveyData_get_rbend_angle_in(survey, i_survey + 1);
        entry.pose = pose_matrix_from_survey(survey, i_survey + 1);
    }
    else {
        // Properly interpolate between the current and the next survey entry
        const float_type t = (s_target - s_current) / (s_next - s_current);
        entry.angle = t * SurveyData_get_angle(survey, i_survey);
        entry.length = t * SurveyData_get_length(survey, i_survey);
        entry.s = s_current + entry.length;

        const Pose pose_current = pose_matrix_from_survey(survey, i_survey);
        if (isfinite(entry.rbend_angle_in) && fabs(SurveyData_get_angle(survey, i_survey)) >= eps) {
            /*
                This is a special case for a straight-body RBend.
                Enter using its entrance rotation and axis shift, then advance by the corresponding body length.
            */
            const float_type full_angle = SurveyData_get_angle(survey, i_survey);
            const float_type full_length = SurveyData_get_length(survey, i_survey);
            const float_type angle_in = entry.rbend_angle_in;
            const float_type angle_out = full_angle - angle_in;
            const float_type length_straight = full_length / full_angle * (sin(angle_in) + sin(angle_out));
            const float_type shift_x_in = entry.rbend_shift_x_in;

            const Pose entry_rotation = survey_arc_matrix(0, angle_in, entry.tilt);
            const Pose body_transform = transform_to_matrix((Transform) {
                .x = -shift_x_in * cos(entry.tilt),
                .y = -shift_x_in * sin(entry.tilt),
                .s = t * length_straight,
            });
            entry.pose = matrix_multiply(
                matrix_multiply(pose_current, entry_rotation),
                body_transform
            );
        }
        else {
            const Pose tilted_arc = survey_arc_matrix(entry.length, entry.angle, entry.tilt);
            entry.pose = matrix_multiply(pose_current, tilted_arc);
        }
    }

    return entry;
}


static inline SurveyEntry_s survey_entry_nan(void)
{
    SurveyEntry_s entry;
    entry.s = NAN;
    entry.angle = NAN;
    entry.length = NAN;
    entry.tilt = NAN;
    entry.rbend_shift_x_in = NAN;
    entry.rbend_angle_in = NAN;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            entry.pose.mat[i][j] = NAN;
        }
    }
    return entry;
}


static inline void write_survey_entry(
    const SurveyData sliced,
    const uint32_t i_sliced,
    const SurveyEntry_s entry
)
{
    SurveyData_set_s(sliced, i_sliced, entry.s);
    SurveyData_set_angle(sliced, i_sliced, entry.angle);
    SurveyData_set_length(sliced, i_sliced, entry.length);
    SurveyData_set_tilt(sliced, i_sliced, entry.tilt);
    SurveyData_set_rbend_shift_x_in(sliced, i_sliced, entry.rbend_shift_x_in);
    SurveyData_set_rbend_angle_in(sliced, i_sliced, entry.rbend_angle_in);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            SurveyData_set_pose(sliced, i_sliced, i, j, entry.pose.mat[i][j]);
        }
    }
}


void resample_survey_table(
    const SurveyData survey,
    const float_type* const s,
    const SurveyData sliced
)
/*
    Interpolate the survey table at position ``s`` and store the result in ``sliced``.
    All provided inputs/outputs are expected to be sorted by the s coordinate.

    Parameters:
    -----------
    survey:
        Survey data containing the original table.
    s:
        Position along the beamline for which to interpolate the survey data.
    sliced:
        Output array where the interpolated survey data will be stored. Should
        have enough space to hold the interpolated values (len(s) entries).
*/
{
    const float_type eps = APER_PRECISION;
    const uint32_t survey_len = SurveyData_len_s(survey);
    const uint32_t sliced_len = SurveyData_len_s(sliced);
    const float_type s_first = SurveyData_get_s(survey, 0);
    const float_type s_last = SurveyData_get_s(survey, survey_len - 1);

    uint32_t i_survey = 0;

    for (uint32_t i_sliced = 0; i_sliced < sliced_len; i_sliced++) {
        const float_type s_target = s[i_sliced];

        // If out of bounds, write NANs
        if (s_target < s_first - eps || s_target > s_last + eps) {
            write_survey_entry(sliced, i_sliced, survey_entry_nan());
            continue;
        }

        // If s is basically at the start, take the starting survey point
        if (s_target <= s_first + eps) {
            write_survey_entry(sliced, i_sliced, interpolate_survey_table_entry(survey, s_first, 0));
            continue;
        }

        // If s is basically at the end, take the end survey point
        if (s_target >= s_last - eps) {
            write_survey_entry(sliced, i_sliced, interpolate_survey_table_entry(survey, s_last, survey_len - 2));
            continue;
        }

        // Fast-forward through survey to find the relevant survey segment for the s_target
        while (i_survey < survey_len - 2) {
            const float_type s_current = SurveyData_get_s(survey, i_survey);
            const float_type s_next = SurveyData_get_s(survey, i_survey + 1);
            if (s_current <= s_target && s_target <= s_next) {
                break;
            }
            i_survey++;
        }

        // Interpolate along the survey segment
        write_survey_entry(sliced, i_sliced, interpolate_survey_table_entry(survey, s_target, i_survey));
    }
}

#endif /* SURVEY_TOOLS_H */
