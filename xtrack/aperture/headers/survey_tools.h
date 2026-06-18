#ifndef SURVEY_TOOLS_H
#define SURVEY_TOOLS_H

#include "base.h"
#include "segment3d.h"


typedef struct {
    float_type s;       // position along the beamline}
    float_type angle;   // angle of the pose vector
    float_type length;  // length of the segment
    float_type tilt;    // tilt of the segment
    Pose pose; // pose matrix (4x4) for each entry
} SurveyEntry_s;


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


static inline Point3D survey_point(SurveyData survey, uint32_t idx) {
    Pose pose = pose_matrix_from_survey(survey, idx);
    return (Point3D) {
        .x = pose.mat[0][3],
        .y = pose.mat[1][3],
        .z = pose.mat[2][3]
    };
}


static inline LineSegment3D survey_line_segment(SurveyData survey, uint32_t idx) {
    const Point3D entry = survey_point(survey, idx);
    const Point3D exit = survey_point(survey, idx + 1);
    return (LineSegment3D) {
        .start = entry,
        .end = exit
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


static inline SurveyEntry_s survey_entry_nan(void)
{
    SurveyEntry_s entry;
    entry.s = NAN;
    entry.angle = NAN;
    entry.length = NAN;
    entry.tilt = NAN;
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
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            SurveyData_set_pose(sliced, i_sliced, i, j, entry.pose.mat[i][j]);
        }
    }
}


static inline uint32_t find_survey_segment_for_s(
    const SurveyData survey,
    const float_type s_target
)
{
    const uint32_t survey_len = SurveyData_len_s(survey);
    uint32_t left = 0;
    uint32_t right = survey_len - 2;

    while (left < right) {
        const uint32_t mid = left + (right - left + 1) / 2;
        if (SurveyData_get_s(survey, mid) <= s_target) {
            left = mid;
        } else {
            right = mid - 1;
        }
    }

    return left;
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
    uint8_t i_survey_initialized = 0;
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

        if (!i_survey_initialized || s_target < SurveyData_get_s(survey, i_survey)) {
            i_survey = find_survey_segment_for_s(survey, s_target);
            i_survey_initialized = 1;
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
