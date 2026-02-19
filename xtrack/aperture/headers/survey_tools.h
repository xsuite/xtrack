#ifndef SURVEY_TOOLS_H
#define SURVEY_TOOLS_H

#include "base.h"


typedef struct {
    float_type s;       // position along the beamline}
    float_type angle;   // angle of the tangent vector
    float_type length;  // length of the segment
    float_type tilt;    // tilt of the segment
    float_type tangent[4][4]; // tangent matrix (4x4) for each entry
} SurveyEntry_s;


SurveyEntry_s interpolate_survey_table_entry(
    const SurveyData survey,
    const float_type s_target,
    const uint32_t i_survey
)
{
    static const float_type eps = 1e-8f;
    const float_type s_current = SurveyData_get_s(survey, i_survey);
    const float_type s_next = SurveyData_get_s(survey, i_survey + 1);

    SurveyEntry_s entry;
    entry.tilt = SurveyData_get_tilt(survey, i_survey);

    if (fabs(s_target - s_current) < eps) {
        // Simply copy the current survey entry
        entry.s = SurveyData_get_s(survey, i_survey);
        entry.angle = SurveyData_get_angle(survey, i_survey);
        entry.length = SurveyData_get_length(survey, i_survey);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                entry.tangent[i][j] = SurveyData_get_tangent(survey, i_survey, i, j);
            }
        }
    }
    else {
        // Properly interpolate between the current and the next survey entry
        const float_type t = (s_target - s_current) / (s_next - s_current);
        entry.angle = t * SurveyData_get_angle(survey, i_survey);
        entry.length = t * SurveyData_get_length(survey, i_survey);
        entry.s = s_current + entry.length;

        float_type tangent_current[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                tangent_current[i][j] = SurveyData_get_tangent(survey, i_survey, i, j);
            }
        }

        float_type ct = cos(entry.tilt), st = sin(entry.tilt);
        float_type ca  = cos(entry.angle), sa = sin(entry.angle);

        float_type dx = -entry.length * sinc(entry.angle * 0.5f) * sin(entry.angle * 0.5f);
        float_type ds = entry.length * sinc(entry.angle);

        float_type tilted_arc[4][4] = {
            {ct * ca,  -st, -ct * sa,  ct * dx },
            {st * ca,   ct, -st * sa,  st * dx },
            {     sa,  0.f,       ca,       ds },
            {    0.f,  0.f,      0.f,      1.f }
        };

        matrix_multiply_4x4(tangent_current, tilted_arc, entry.tangent);
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
                    SurveyData_set_tangent(sliced, i_sliced, i, j, entry.tangent[i][j]);
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