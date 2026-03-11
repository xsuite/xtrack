#ifndef SURVEY_TOOLS_H
#define SURVEY_TOOLS_H

#include "base.h"
#include "path3d.h"


typedef struct {
    float_type s;       // position along the beamline}
    float_type angle;   // angle of the pose vector
    float_type length;  // length of the segment
    float_type tilt;    // tilt of the segment
    Pose pose; // pose matrix (4x4) for each entry
} SurveyEntry_s;


inline float_type get_survey_max_s(const SurveyData survey)
/* Get the max s of the survey. */
{
    // Rely on the fact that the last element is `_end_point` (length = 0).
    // This needs to change, if that is ever not the case.
    const uint32_t survey_num_entries = SurveyData_len_s(survey);
    return SurveyData_get_s(survey, survey_num_entries - 1);
}


inline Pose pose_matrix_from_survey(const SurveyData survey, const uint32_t idx) {
    Pose m;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            m.mat[i][j] = SurveyData_get_pose(survey, idx, i, j);
        }
    }
    return m;
}


inline Point3D survey_point(SurveyData survey, uint32_t idx) {
    Pose pose = pose_matrix_from_survey(survey, idx);
    return (Point3D) {
        .x = pose.mat[0][3],
        .y = pose.mat[1][3],
        .z = pose.mat[2][3]
    };
}


inline LineSegment3D survey_line_segment(SurveyData survey, uint32_t idx) {
    const Point3D entry = survey_point(survey, idx);
    const Point3D exit = survey_point(survey, idx + 1);
    return (LineSegment3D) {
        .start = entry,
        .end = exit
    };
}


inline ArcSegment3D survey_arc_segment(SurveyData survey, uint32_t idx) {
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


inline Segment3D survey_segment(SurveyData survey, uint32_t idx) {
    const float_type angle = SurveyData_get_angle(survey, idx);
    const float_type length = SurveyData_get_length(survey, idx);

    if (fabs(angle) < APER_PRECISION || fabs(length) < APER_PRECISION)
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

    if (fabs(s_target - s_current) < eps) {
        // Simply copy the current survey entry
        entry.s = SurveyData_get_s(survey, i_survey);
        entry.angle = SurveyData_get_angle(survey, i_survey);
        entry.length = SurveyData_get_length(survey, i_survey);
        entry.pose = pose_matrix_from_survey(survey, i_survey);
    }
    else {
        // Properly interpolate between the current and the next survey entry
        const float_type t = (s_target - s_current) / (s_next - s_current);
        entry.angle = t * SurveyData_get_angle(survey, i_survey);
        entry.length = t * SurveyData_get_length(survey, i_survey);
        entry.s = s_current + entry.length;

        const Pose pose_current = pose_matrix_from_survey(survey, i_survey);
        const Pose tilted_arc = arc_matrix(entry.length, entry.angle, 0);
        entry.pose = matrix_multiply(pose_current, tilted_arc);
    }

    return entry;
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
    const uint32_t survey_len = SurveyData_len_s(survey);
    const uint32_t sliced_len = SurveyData_len_s(sliced);

    uint32_t i_sliced = 0;
    uint32_t i_survey = 0;

    while (i_sliced < sliced_len && i_survey < survey_len - 1) {
        const float_type s_current = SurveyData_get_s(survey, i_survey);
        const float_type s_next = SurveyData_get_s(survey, i_survey + 1);

        if (s_current <= s[i_sliced] && s[i_sliced] <= s_next) {
            // If we're in the right interval, interpolate
            SurveyEntry_s entry = interpolate_survey_table_entry(
                survey, s[i_sliced], i_survey);

            SurveyData_set_s(sliced, i_sliced, entry.s);
            SurveyData_set_angle(sliced, i_sliced, entry.angle);
            SurveyData_set_length(sliced, i_sliced, entry.length);
            SurveyData_set_tilt(sliced, i_sliced, entry.tilt);
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    SurveyData_set_pose(sliced, i_sliced, i, j, entry.pose.mat[i][j]);
                }
            }
            i_sliced++;
        }
        else {
            // If not the right interval, wait for it to come
            i_survey++;
        }
    }
}

#endif /* SURVEY_TOOLS_H */
