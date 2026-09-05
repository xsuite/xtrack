// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_INTEGRATOR_H
#define XTRACK_INTEGRATOR_H

/*
    Single shared implementation of the slice/kick integration schemes
    (TEAPOT, uniform, Yoshida 2/4/6/8), used by both track_magnet.h and
    track_rf.h.

    DRIFT_FUNCTION and KICK_FUNCTION are expected to be macros of the form
    FUNCTION(PART, WEIGHT), and RADIATION_MACRO a macro of the form
    MACRO(LENGTH, CODE) wrapping CODE (which may be empty of any radiation
    handling, see WITH_RADIATION/WITH_RF_RADIATION at the call sites).

    Function pointers are avoided on purpose: OpenCL C does not support
    them, so the integrator is expressed as a macro that is expanded
    in-place at each call site instead of being a genuine callable function.
*/
#define RUN_INTEGRATOR(INTEGRATOR, LENGTH, NUM_KICKS, PART, \
                        DRIFT_FUNCTION, KICK_FUNCTION, RADIATION_MACRO) \
do { \
    if ((INTEGRATOR) == 1){ /* TEAPOT */ \
\
        RADIATION_MACRO((LENGTH), \
            const double kick_weight = 1. / (NUM_KICKS); \
            double edge_drift_weight = 0.5; \
            double inside_drift_weight = 0; \
            if ((NUM_KICKS) > 1) { \
                edge_drift_weight = 1. / (2 * (1 + (NUM_KICKS))); \
                inside_drift_weight = ( \
                    ((double) (NUM_KICKS)) \
                        / ((double)((NUM_KICKS)*(NUM_KICKS)) - 1)); \
            } \
\
            DRIFT_FUNCTION((PART), edge_drift_weight*(LENGTH)); \
            for (int i_kick=0; i_kick<(NUM_KICKS) - 1; i_kick++) { \
                KICK_FUNCTION((PART), kick_weight); \
                DRIFT_FUNCTION((PART), inside_drift_weight*(LENGTH)); \
            } \
            KICK_FUNCTION((PART), kick_weight); \
            DRIFT_FUNCTION((PART), edge_drift_weight*(LENGTH)); \
        ) \
\
    } \
    else if ((INTEGRATOR) == 3){ /* uniform */ \
\
        const double kick_weight = 1. / (NUM_KICKS); \
        const double drift_weight = kick_weight; \
\
        for (int i_kick=0; i_kick<(NUM_KICKS); i_kick++) { \
            RADIATION_MACRO(drift_weight*(LENGTH), \
                DRIFT_FUNCTION((PART), 0.5*drift_weight*(LENGTH)); \
                KICK_FUNCTION((PART), kick_weight); \
                DRIFT_FUNCTION((PART), 0.5*drift_weight*(LENGTH)); \
            ) \
        } \
\
    } \
    else if ((INTEGRATOR) == 2) { /* YOSHIDA 6 */ \
\
        const int64_t n_kicks_yoshida = 7; \
        const int64_t num_slices = ((NUM_KICKS) / n_kicks_yoshida \
                                + ((NUM_KICKS) % n_kicks_yoshida != 0)); \
        const double slice_length = (LENGTH) / num_slices; \
        const double kick_weight = 1. / num_slices; \
        /* Yoshida90 coefficients, as tabulated and selected by MAD-NG. */ \
        const double d_yoshida[] = {0.39225680523877998, 0.51004341191845848, -0.47105338540975655, 0.068753168252518093, 0.068753168252518093, -0.47105338540975655, 0.51004341191845848, 0.39225680523877998}; \
        const double k_yoshida[] = {0.78451361047755996, 0.23557321335935699, -1.1776799841788701, 1.3151863206839063, -1.1776799841788701, 0.23557321335935699, 0.78451361047755996}; \
\
        for (int64_t i_slice = 0; i_slice < num_slices; i_slice++) { \
            RADIATION_MACRO(slice_length, \
                DRIFT_FUNCTION((PART), slice_length * d_yoshida[0]); \
                for (int64_t i_kick = 0; i_kick < n_kicks_yoshida; i_kick++) { \
                    KICK_FUNCTION((PART), kick_weight * k_yoshida[i_kick]); \
                    DRIFT_FUNCTION((PART), slice_length * d_yoshida[i_kick + 1]); \
                } \
            ) \
        } \
    } \
    else if ((INTEGRATOR) == 4) { /* YOSHIDA 4 */ \
\
        const int64_t n_kicks_yoshida = 3; \
        const int64_t num_slices = ((NUM_KICKS) / n_kicks_yoshida \
                                + ((NUM_KICKS) % n_kicks_yoshida != 0)); \
        const double slice_length = (LENGTH) / num_slices; \
        const double kick_weight = 1. / num_slices; \
        /* Yoshida90 coefficients, as tabulated and selected by MAD-NG. */ \
        const double d_yoshida[] = {0.67560359597982889, -0.17560359597982889, -0.17560359597982889, 0.67560359597982889}; \
        const double k_yoshida[] = {1.3512071919596578, -1.7024143839193155, 1.3512071919596578}; \
\
        for (int64_t i_slice = 0; i_slice < num_slices; i_slice++) { \
            RADIATION_MACRO(slice_length, \
                DRIFT_FUNCTION((PART), slice_length * d_yoshida[0]); \
                for (int64_t i_kick = 0; i_kick < n_kicks_yoshida; i_kick++) { \
                    KICK_FUNCTION((PART), kick_weight * k_yoshida[i_kick]); \
                    DRIFT_FUNCTION((PART), slice_length * d_yoshida[i_kick + 1]); \
                } \
            ) \
        } \
    } \
    else if ((INTEGRATOR) == 5) { /* YOSHIDA 8 */ \
\
        const int64_t n_kicks_yoshida = 15; \
        const int64_t num_slices = ((NUM_KICKS) / n_kicks_yoshida \
                                + ((NUM_KICKS) % n_kicks_yoshida != 0)); \
        const double slice_length = (LENGTH) / num_slices; \
        const double kick_weight = 1. / num_slices; \
        /* Yoshida90 coefficients, as tabulated and selected by MAD-NG. */ \
        const double d_yoshida[] = {0.45742212311487002, 0.58426879139798449, -0.59557945014712543, -0.80154643611436149, 0.88994925112725842, -0.011235547676365032, -0.92890519179175246, 0.90562646008949144, 0.90562646008949144, -0.92890519179175246, -0.011235547676365032, 0.88994925112725842, -0.80154643611436149, -0.59557945014712543, 0.58426879139798449, 0.45742212311487002}; \
        const double k_yoshida[] = {0.91484424622974003, 0.253693336566229, -1.4448522368604799, -0.15824063536824301, 1.9381391376227599, -1.96061023297549, 0.102799849391985, 1.7084530707869978, 0.102799849391985, -1.96061023297549, 1.9381391376227599, -0.15824063536824301, -1.4448522368604799, 0.253693336566229, 0.91484424622974003}; \
\
        for (int64_t i_slice = 0; i_slice < num_slices; i_slice++) { \
            RADIATION_MACRO(slice_length, \
                DRIFT_FUNCTION((PART), slice_length * d_yoshida[0]); \
                for (int64_t i_kick = 0; i_kick < n_kicks_yoshida; i_kick++) { \
                    KICK_FUNCTION((PART), kick_weight * k_yoshida[i_kick]); \
                    DRIFT_FUNCTION((PART), slice_length * d_yoshida[i_kick + 1]); \
                } \
            ) \
        } \
    } \
} while(0)

#endif
