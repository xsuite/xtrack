// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_TRACK_MISALIGNMENT_H
#define XTRACK_TRACK_MISALIGNMENT_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_xrotation.h>
#include <beam_elements/elements_src/track_yrotation.h>
#include <beam_elements/elements_src/track_srotation.h>
#include <beam_elements/elements_src/track_drift.h>
#include <beam_elements/elements_src/track_xyshift.h>


#define IF_NONZERO(VALUE, EXPRESSION) { if (VALUE != 0.0) { EXPRESSION; } }
#define LOOP_PARTICLES(PART0, CODE) {\
    START_PER_PARTICLE_BLOCK(PART0, part) { \
        CODE ; \
    } END_PER_PARTICLE_BLOCK; }
#define Y_ROTATE(PART0, THETA) IF_NONZERO(THETA, LOOP_PARTICLES(PART0, YRotation_single_particle(part, sin(THETA), cos(THETA), tan(THETA))))
// Flip some signs so that the input matches the MAD-X survey convention
#define X_ROTATE(PART0, PHI) IF_NONZERO(PHI, LOOP_PARTICLES(PART0, XRotation_single_particle(part, -sin(PHI), cos(PHI), -tan(PHI))))
#define S_ROTATE(PART0, PSI) IF_NONZERO(PSI, LOOP_PARTICLES(PART0, SRotation_single_particle(part, sin(PSI), cos(PSI))))
#define XY_SHIFT(PART0, DX, DY) LOOP_PARTICLES(PART0, XYShift_single_particle(part, DX, DY))
#define S_SHIFT(PART0, DS) IF_NONZERO(DS, LOOP_PARTICLES(PART0, { \
        Drift_single_particle_exact(part, DS); \
        LocalParticle_add_to_zeta(part, -DS); \
        LocalParticle_add_to_s(part, -DS); \
    }))


GPUFUN void matrix_multiply_4x4(const double[4][4], const double[4][4], double[4][4]);
GPUFUN void matrix_rigid_affine_inverse(const double[4][4], double[4][4]);


GPUFUN
void track_misalignment_entry_straight(
    LocalParticle* part0,  // LocalParticle to track
    double dx,  // misalignment in x
    double dy,  // misalignment in y
    double ds,  // misalignment in s
    double theta, // rotation around y, yaw, positive s to x
    double phi,  // rotation around x, pitch, positive s to y
    double psi_no_frame,  // rotation around s, roll, positive y to x
    double anchor, // anchor of the misalignment as offset in m from entry
    double length,  // length of the misaligned element
    double psi_with_frame,  // psi_with_frame of the element, positive s to x
    int8_t backtrack
) {
    // Silence the warning about unused variable length
    (void)length; // kept for API consistency with track_misalignment_exit_straight

    const double mis_x = dx - anchor * cos(phi) * sin(theta);
    const double mis_y = dy - anchor * sin(phi);
    const double mis_s = ds - anchor * (cos(phi) * cos(theta) - 1);

    // Apply transformations
    if (!backtrack){
        XY_SHIFT(part0, mis_x, mis_y);
        S_SHIFT(part0, mis_s);
        Y_ROTATE(part0, theta);
        X_ROTATE(part0, phi);
        S_ROTATE(part0, psi_no_frame);
        S_ROTATE(part0, psi_with_frame);
    } else {
        S_ROTATE(part0, -psi_with_frame);
        S_ROTATE(part0, -psi_no_frame);
        X_ROTATE(part0, -phi);
        Y_ROTATE(part0, -theta);
        S_SHIFT(part0, -mis_s);
        XY_SHIFT(part0, -mis_x, -mis_y);
    }
}


GPUFUN
void track_misalignment_exit_straight(
    LocalParticle* part0,  // LocalParticle to track
    double dx,  // misalignment in x
    double dy,  // misalignment in y
    double ds,  // misalignment in s
    double theta, // rotation around y, yaw, positive s to x
    double phi,  // rotation around x, pitch, positive s to y
    double psi_no_frame,  // rotation around s, roll, positive y to x
    double anchor, // anchor of the misalignment as offset in m from entry
    double length,  // length of the misaligned element
    double psi_with_frame,  // psi_with_frame of the element, positive s to x
    int8_t backtrack
) {
    const double neg_part_length = anchor - length;
    const double mis_x = neg_part_length * cos(phi) * sin(theta) - dx;
    const double mis_y = neg_part_length * sin(phi) - dy;
    const double mis_s = neg_part_length * (cos(phi) * cos(theta) - 1) - ds;

    // Apply transformations
    if (!backtrack){
        S_ROTATE(part0, -psi_with_frame);
        S_ROTATE(part0, -psi_no_frame);
        X_ROTATE(part0, -phi);
        Y_ROTATE(part0, -theta);
        S_SHIFT(part0, mis_s);
        XY_SHIFT(part0, mis_x, mis_y);
    } else {
        XY_SHIFT(part0, -mis_x, -mis_y);
        S_SHIFT(part0, -mis_s);
        Y_ROTATE(part0, theta);
        X_ROTATE(part0, phi);
        S_ROTATE(part0, psi_no_frame);
        S_ROTATE(part0, psi_with_frame);
    }
}


GPUFUN
void track_misalignment_entry_curved(
    LocalParticle* part0,  // LocalParticle to track
    double dx,  // misalignment in x
    double dy,  // misalignment in y
    double ds,  // misalignment in s
    double theta, // rotation around y, yaw, positive s to x
    double phi,  // rotation around x, pitch, positive s to y
    double psi_no_frame,  // rotation around s, roll, positive y to x
    double anchor, // anchor of the misalignment as offset in m from entry
    double length,  // length of the misaligned element
    double angle,  // angle by which the element bends the reference frame
    double psi_with_frame,  // psi_with_frame of the element, positive s to x
    int8_t backtrack
) {
    if (angle == 0.0) {
        track_misalignment_entry_straight(part0, dx, dy, ds, theta, phi,
            psi_no_frame, anchor, length, psi_with_frame, backtrack);
        return;
    }
    // Precompute trigonometric functions
    const double s_phi = sin(phi), c_phi = cos(phi);
    const double s_theta = sin(theta), c_theta = cos(theta);
    const double s_psi = sin(psi_no_frame), c_psi = cos(psi_no_frame);

    /* We need to compute the transformation that takes us from the aligned
       frame to the entry of the element in the misaligned frame:

       misaligned_entry = matrix_first_part @ misalignment_matrix @ inv(matrix_first_part)
       
       where:
       - misalignment_matrix is the matrix that applies the misalignment
       - matrix_first_part is the matrix that takes us from the entry of the
         element to the anchor of the misalignment
    */

    // Misalignment matrix
    const double misalignment_matrix[4][4] = {
        {
            -s_phi * s_psi * s_theta + c_psi * c_theta,
            -c_psi * s_phi * s_theta - c_theta * s_psi,
            c_phi * s_theta,
            dx
        },
        {
            c_phi * s_psi,
            c_phi * c_psi,
            s_phi, dy
        },
        {
            -c_theta * s_phi * s_psi - c_psi * s_theta,
            -c_psi * c_theta * s_phi + s_psi * s_theta,
            c_phi * c_theta,
            ds
        },
        {0, 0, 0, 1}
    };

    // Compute matrix that takes us from the reference point of the misalignment
    // to the entry of the element
    double anchor_frac = length == 0.0 ? 0 : anchor / length;
    const double part_angle = angle * anchor_frac;
    const double rho = length / angle;
    const double delta_x_first_part = rho * (cos(part_angle) - 1) * cos(psi_with_frame);
    const double delta_y_first_part = rho * (cos(part_angle) - 1) * sin(psi_with_frame);
    const double delta_s_first_part = rho * sin(part_angle);

    const double matrix_first_part[4][4] = {
            {
                (cos(part_angle) - 1) * POW2(cos(psi_with_frame)) + 1,
                (cos(part_angle) - 1) * cos(psi_with_frame) * sin(psi_with_frame),
                -cos(psi_with_frame) * sin(part_angle),
                delta_x_first_part
            },
            {
                (cos(part_angle) - 1) * cos(psi_with_frame) * sin(psi_with_frame),
                (cos(part_angle) - 1) * POW2(sin(psi_with_frame)) + 1,
                -sin(part_angle) * sin(psi_with_frame),
                delta_y_first_part
            },
            {
                cos(psi_with_frame) * sin(part_angle),
                sin(part_angle) * sin(psi_with_frame),
                cos(part_angle),
                delta_s_first_part
            },
            {0, 0, 0, 1}
        };

    double inv_matrix_first_part[4][4];
    matrix_rigid_affine_inverse(matrix_first_part, inv_matrix_first_part);

    // Compute the transformation that takes us from the aligned frame to the
    // entry of the element in the misaligned frame
    double misaligned_entry[4][4], temp[4][4];
    matrix_multiply_4x4(matrix_first_part, misalignment_matrix, temp);
    matrix_multiply_4x4(temp, inv_matrix_first_part, misaligned_entry);

    // Extract the basic transformations from the misalignment matrix
    const double mis_x = misaligned_entry[0][3];
    const double mis_y = misaligned_entry[1][3];
    const double mis_s = misaligned_entry[2][3];

    const double rot_theta = atan2(misaligned_entry[0][2], misaligned_entry[2][2]);
    const double rot_phi = atan2(misaligned_entry[1][2], sqrt(misaligned_entry[1][0] * misaligned_entry[1][0] + misaligned_entry[1][1] * misaligned_entry[1][1]));
    const double rot_psi = atan2(misaligned_entry[1][0], misaligned_entry[1][1]);

    // Apply transformations
    if (!backtrack) {
        XY_SHIFT(part0, mis_x, mis_y);
        S_SHIFT(part0, mis_s);
        Y_ROTATE(part0, rot_theta);
        X_ROTATE(part0, rot_phi);
        S_ROTATE(part0, rot_psi);
        S_ROTATE(part0, psi_with_frame);
    } else {
        S_ROTATE(part0, -psi_with_frame);
        S_ROTATE(part0, -rot_psi);
        X_ROTATE(part0, -rot_phi);
        Y_ROTATE(part0, -rot_theta);
        S_SHIFT(part0, -mis_s);
        XY_SHIFT(part0, -mis_x, -mis_y);
    }
}


GPUFUN
void track_misalignment_exit_curved(
    LocalParticle* part0,  // LocalParticle to track
    double dx,  // misalignment in x
    double dy,  // misalignment in y
    double ds,  // misalignment in s
    double theta, // rotation around y, yaw, positive s to x
    double phi,  // rotation around x, pitch, positive s to y
    double psi_no_frame,  // rotation around s, roll, positive y to x
    double anchor, // anchor of the misalignment as a fraction of the length
    double length,  // length of the misaligned element
    double angle,  // angle by which the element bends the reference frame
    double psi_with_frame,  // psi_with_frame of the element, positive s to x
    int8_t backtrack  // whether to backtrack the particle
) {
    if (angle == 0.0) {
        track_misalignment_exit_straight(
            part0, dx, dy, ds, theta, phi, psi_no_frame, anchor, length,
            psi_with_frame, backtrack);
        return;
    }
    // Precompute trigonometric functions
    double s_phi = sin(phi), c_phi = cos(phi);
    double s_theta = sin(theta), c_theta = cos(theta);
    double s_psi = sin(psi_no_frame), c_psi = cos(psi_no_frame);

    /* We need to compute the transformation that takes us from the misaligned
       frame to the aligned frame:

       realign = inv(matrix_second_part) * inv(misalignment_matrix) * matrix_second_part

       where:
       - misalignment_matrix is the matrix that applies the misalignment
       - matrix_second_part is the matrix that takes us from the frame in the
         middle of the element (anchor) to the end of the element
    */

    // Misalignment matrix
    const double misalignment_matrix[4][4] = {
        {
            -s_phi * s_psi * s_theta + c_psi * c_theta,
            -c_psi * s_phi * s_theta - c_theta * s_psi,
            c_phi * s_theta,
            dx
        },
        {
            c_phi * s_psi,
            c_phi * c_psi,
            s_phi,
            dy
        },
        {
            -c_theta * s_phi * s_psi - c_psi * s_theta,
            -c_psi * c_theta * s_phi + s_psi * s_theta,
            c_phi * c_theta,
            ds
        },
        {0, 0, 0, 1}
    };

    // Compute the inverse of the misalignment matrix
    double inv_misalignment_matrix[4][4];
    matrix_rigid_affine_inverse(misalignment_matrix, inv_misalignment_matrix);

    // Compute the inverse of the matrix that takes us from the point of the
    // misalignment to the exit of the element.
    double anchor_frac = length == 0.0 ? 0.0 : anchor / length;
    const double anchor_compl = 1 - anchor_frac;
    const double part_angle = angle * anchor_compl;
    const double rho = length / angle;

    const double s_part_angle = sin(part_angle), c_part_angle = cos(part_angle);
    const double s_tilt = sin(psi_with_frame), c_tilt = cos(psi_with_frame);

    const double delta_x_second_part = rho * (c_part_angle - 1) * c_tilt;
    const double delta_y_second_part = rho * (c_part_angle - 1) * s_tilt;
    const double delta_s_second_part = rho * s_part_angle;

    const double matrix_second_part[4][4] = {
        {
            (c_part_angle - 1) * POW2(c_tilt) + 1,
            (c_part_angle - 1) * c_tilt * s_tilt,
            c_tilt * -s_part_angle,
            delta_x_second_part
        },
        {
            (c_part_angle - 1) * c_tilt * s_tilt,
            (c_part_angle - 1) * POW2(s_tilt) + 1,
            -s_part_angle * s_tilt,
            delta_y_second_part
        },
        {
            c_tilt * s_part_angle,
            s_part_angle * s_tilt,
            c_part_angle,
            delta_s_second_part
        },
        {0, 0, 0, 1}
    };

    double inv_matrix_second_part[4][4];
    matrix_rigid_affine_inverse(matrix_second_part, inv_matrix_second_part);

    // Compute the realignment matrix
    double realign[4][4], temp[4][4];
    matrix_multiply_4x4(inv_matrix_second_part, inv_misalignment_matrix, temp);
    matrix_multiply_4x4(temp, matrix_second_part, realign);

    // Extract the basic transformations from the realignment matrix
    double mis_x = realign[0][3];
    double mis_y = realign[1][3];
    double mis_s = realign[2][3];

    double rot_theta = atan2(realign[0][2], realign[2][2]);
    double rot_phi = atan2(realign[1][2], sqrt(realign[1][0] * realign[1][0] + realign[1][1] * realign[1][1]));
    double rot_psi = atan2(realign[1][0], realign[1][1]);

    // Apply transformations
    if (!backtrack){
        S_ROTATE(part0, -psi_with_frame);
        XY_SHIFT(part0, mis_x, mis_y);
        S_SHIFT(part0, mis_s);
        Y_ROTATE(part0, rot_theta);
        X_ROTATE(part0, rot_phi);
        S_ROTATE(part0, rot_psi);
    } else {
        S_ROTATE(part0, -rot_psi);
        X_ROTATE(part0, -rot_phi);
        Y_ROTATE(part0, -rot_theta);
        S_SHIFT(part0, -mis_s);
        XY_SHIFT(part0, -mis_x, -mis_y);
        S_ROTATE(part0, psi_with_frame);
    }
}


GPUFUN
void matrix_multiply_4x4(const double a[4][4], const double b[4][4], double result[4][4]) {
    // Multiply two 4x4 matrices `a` and `b`, and store the result in `result`.
    for (int i = 0; i < 4; i++) {
        result[i][0] = a[i][0] * b[0][0] + a[i][1] * b[1][0] + a[i][2] * b[2][0] + a[i][3] * b[3][0];
        result[i][1] = a[i][0] * b[0][1] + a[i][1] * b[1][1] + a[i][2] * b[2][1] + a[i][3] * b[3][1];
        result[i][2] = a[i][0] * b[0][2] + a[i][1] * b[1][2] + a[i][2] * b[2][2] + a[i][3] * b[3][2];
        result[i][3] = a[i][0] * b[0][3] + a[i][1] * b[1][3] + a[i][2] * b[2][3] + a[i][3] * b[3][3];
    }
}


GPUFUN
void matrix_rigid_affine_inverse(const double m[4][4], double inv_m[4][4]) {
    // Compute the inverse `inv_m` of a rigid affine transformation matrix `m`.
    const double m00 = m[0][0], m01 = m[0][1], m02 = m[0][2], m03 = m[0][3];
    const double m10 = m[1][0], m11 = m[1][1], m12 = m[1][2], m13 = m[1][3];
    const double m20 = m[2][0], m21 = m[2][1], m22 = m[2][2], m23 = m[2][3];

    // Invert the rotation part of `m`: as it's rigid, it's simply a transpose
    inv_m[0][0] = m00;
    inv_m[0][1] = m10;
    inv_m[0][2] = m20;
    inv_m[1][0] = m01;
    inv_m[1][1] = m11;
    inv_m[1][2] = m21;
    inv_m[2][0] = m02;
    inv_m[2][1] = m12;
    inv_m[2][2] = m22;

    // Compute the translation part, -m[:3, :3]^{-1} @ m[:3, 3]
    inv_m[0][3] = -m00 * m03 - m10 * m13 - m20 * m23;
    inv_m[1][3] = -m01 * m03 - m11 * m13 - m21 * m23;
    inv_m[2][3] = -m02 * m03 - m12 * m13 - m22 * m23;

    // Fill the last row
    inv_m[3][0] = 0;
    inv_m[3][1] = 0;
    inv_m[3][2] = 0;
    inv_m[3][3] = 1;
}

#endif  // XTRACK_TRACK_MISALIGNMENT_H
