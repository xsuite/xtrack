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
#include <beam_elements/elements_src/track_solenoid.h>
#include <beam_elements/elements_src/track_xyshift.h>


// Flip some signs to as the input expects MAD-X convention
#define Y_ROTATE(PART, THETA) YRotation_single_particle(PART, sin(THETA), cos(THETA), tan(THETA))
#define X_ROTATE(PART, PHI) XRotation_single_particle(PART, -sin(PHI), cos(PHI), -tan(PHI))
#define S_ROTATE(PART, PSI) SRotation_single_particle(PART, sin(PSI), cos(PSI))
#define XY_SHIFT(PART, DX, DY) XYShift_single_particle(PART, DX, DY)
#define S_SHIFT(PART, DS) Solenoid_thick_track_single_particle(part, DS, 0, 0)


double sinc(double);
void matrix_multiply_4x4(const double[4][4], const double[4][4], double[4][4]);
void print_matrix_4x4(const char*, const double[4][4]);


void track_misalignment_entry_curved(
    LocalParticle* part0,  // LocalParticle to track
    double dx,  // misalignment in x
    double dy,  // misalignment in y
    double ds,  // misalignment in s
    double theta, // rotation around y, yaw, positive s to x
    double phi,  // rotation around x, pitch, positive s to y
    double psi,  // rotation around s, roll, positive y to x
    double location, // location of the misalignment as a fraction of the length
    double length,  // length of the misaligned element
    double angle  // angle by which the element bends the reference frame
) {
    // Precompute trigonometric functions
    const double s_phi = sin(phi), c_phi = cos(phi);
    const double s_theta = sin(theta), c_theta = cos(theta);
    const double s_psi = sin(psi), c_psi = cos(psi);

    /* We need to compute the transformation that takes us from the aligned
       frame to the entry of the element in the misaligned frame:

       misaligned_entry = matrix_first_part @ misalignment_matrix @ inv(matrix_first_part)
       
       where:
       - misalignment_matrix is the matrix that applies the misalignment
       - matrix_first_part is the matrix that takes us from the entry of the
         element to the anchor of the misalignment (s-position of location * length)
    */

    // Misalignment matrix
    const double misalignment_matrix[4][4] = {
        {-s_phi * s_psi * s_theta + c_psi * c_theta, -c_psi * s_phi * s_theta - c_theta * s_psi, c_phi * s_theta, dx},
        {c_phi * s_psi, c_phi * c_psi, s_phi, dy},
        {-c_theta * s_phi * s_psi - c_psi * s_theta, -c_psi * c_theta * s_phi + s_psi * s_theta, c_phi * c_theta, ds},
        {0, 0, 0, 1}
    };

    // Compute matrix that takes us from the reference point of the misalignment
    // to the entry of the element
    const double part_angle = angle * location;
    const double part_length = length * location;
    const double delta_x_first_part = -part_length * sinc(part_angle / 2) * sin(part_angle / 2);
    const double delta_s_first_part = part_length * sinc(part_angle);

    const double matrix_first_part[4][4] = {
        {cos(part_angle), 0, -sin(part_angle), delta_x_first_part},
        {0, 1, 0, 0},
        {sin(part_angle), 0, cos(part_angle), delta_s_first_part},
        {0, 0, 0, 1}
    };

    const double inv_matrix_first_part[4][4] = {
        {cos(part_angle), 0, sin(part_angle), delta_x_first_part},
        {0, 1, 0, 0},
        {-sin(part_angle), 0, cos(part_angle), -delta_s_first_part},
        {0, 0, 0, 1}
    };

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
    START_PER_PARTICLE_BLOCK(part0, part);
        XY_SHIFT(part, mis_x, mis_y);
        S_SHIFT(part, mis_s);
        Y_ROTATE(part, rot_theta);
        X_ROTATE(part, rot_phi);
        S_ROTATE(part, rot_psi);
    END_PER_PARTICLE_BLOCK;
}


void track_misalignment_exit_curved(
    LocalParticle* part0,  // LocalParticle to track
    double dx,  // misalignment in x
    double dy,  // misalignment in y
    double ds,  // misalignment in s
    double theta, // rotation around y, yaw, positive s to x
    double phi,  // rotation around x, pitch, positive s to y
    double psi,  // rotation around s, roll, positive y to x
    double location, // location of the misalignment as a fraction of the length
    double length,  // length of the misaligned element
    double angle  // angle by which the element bends the reference frame
) {
    // Precompute trigonometric functions
    double s_phi = sin(phi), c_phi = cos(phi);
    double s_theta = sin(theta), c_theta = cos(theta);
    double s_psi = sin(psi), c_psi = cos(psi);

    /* We need to compute the transformation that takes us from the misaligned
       frame to the aligned frame:

       realign = inv(matrix_second_part) * inv(misalignment_matrix) * matrix_second_part

       where:
       - misalignment_matrix is the matrix that applies the misalignment
       - matrix_second_part is the matrix that takes us from the frame in the
         middle of the element (location * length) to the end of the element
    */

    // Misalignment matrix
    const double misalignment_matrix[4][4] = {
        {-s_phi * s_psi * s_theta + c_psi * c_theta, -c_psi * s_phi * s_theta - c_theta * s_psi, c_phi * s_theta, dx},
        {c_phi * s_psi, c_phi * c_psi, s_phi, dy},
        {-c_theta * s_phi * s_psi - c_psi * s_theta, -c_psi * c_theta * s_phi + s_psi * s_theta, c_phi * c_theta, ds},
        {0, 0, 0, 1}
    };

    // Compute the inverse of the misalignment matrix
    const double m00 = misalignment_matrix[0][0], m01 = misalignment_matrix[0][1], m02 = misalignment_matrix[0][2], m03 = misalignment_matrix[0][3];
    const double m10 = misalignment_matrix[1][0], m11 = misalignment_matrix[1][1], m12 = misalignment_matrix[1][2], m13 = misalignment_matrix[1][3];
    const double m20 = misalignment_matrix[2][0], m21 = misalignment_matrix[2][1], m22 = misalignment_matrix[2][2], m23 = misalignment_matrix[2][3];

    const double inv_misalignment_matrix[4][4] = {
        {m00, m10, m20, -m00 * m03 - m10 * m13 - m20 * m23},
        {m01, m11, m21, -m01 * m03 - m11 * m13 - m21 * m23},
        {m02, m12, m22, -m02 * m03 - m12 * m13 - m22 * m23},
        {0, 0, 0, 1}
    };

    // Compute the inverse of the matrix that takes us from the point of the
    // misalignment to the exit of the element.
    const double location_compl = 1 - location;
    const double part_angle = angle * location_compl;
    const double part_length = length * location_compl;
    const double delta_x_second_part = -part_length * sinc(part_angle / 2) * sin(part_angle / 2);
    const double delta_s_second_part = part_length * sinc(part_angle);

    const double matrix_second_part[4][4] = {
        {cos(part_angle), 0, -sin(part_angle), delta_x_second_part},
        {0, 1, 0, 0},
        {sin(part_angle), 0, cos(part_angle), delta_s_second_part},
        {0, 0, 0, 1}
    };

    const double inv_matrix_second_part[4][4] = {
        {cos(part_angle), 0, sin(part_angle), delta_x_second_part},
        {0, 1, 0, 0},
        {-sin(part_angle), 0, cos(part_angle), -delta_s_second_part},
        {0, 0, 0, 1}
    };

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
    START_PER_PARTICLE_BLOCK(part0, part);
        XY_SHIFT(part, mis_x, mis_y);
        S_SHIFT(part, mis_s);
        Y_ROTATE(part, rot_theta);
        X_ROTATE(part, rot_phi);
        S_ROTATE(part, rot_psi);
    END_PER_PARTICLE_BLOCK;
}


void track_misalignment_entry_straight(
    LocalParticle* part0,  // LocalParticle to track
    double dx,  // misalignment in x
    double dy,  // misalignment in y
    double ds,  // misalignment in s
    double theta, // rotation around y, yaw, positive s to x
    double phi,  // rotation around x, pitch, positive s to y
    double psi,  // rotation around s, roll, positive y to x
    double location, // location of the misalignment as a fraction of the length
    double length  // length of the misaligned element
) {
    const double part_length = location * length;
    const double mis_x = dx - part_length * cos(phi) * sin(theta);
    const double mis_y = dy - part_length * sin(phi);
    const double mis_s = ds - part_length * (cos(phi) * cos(theta) - 1);

    // Apply transformations
    START_PER_PARTICLE_BLOCK(part0, part);
        XY_SHIFT(part, mis_x, mis_y);
        S_SHIFT(part, mis_s);
        Y_ROTATE(part, theta);
        X_ROTATE(part, phi);
        S_ROTATE(part, psi);
    END_PER_PARTICLE_BLOCK;
}


void track_misalignment_exit_straight(
    LocalParticle* part0,  // LocalParticle to track
    double dx,  // misalignment in x
    double dy,  // misalignment in y
    double ds,  // misalignment in s
    double theta, // rotation around y, yaw, positive s to x
    double phi,  // rotation around x, pitch, positive s to y
    double psi,  // rotation around s, roll, positive y to x
    double location, // location of the misalignment as a fraction of the length
    double length  // length of the misaligned element
) {
    const double part_length = (location - 1) * length;
    const double mis_x = part_length * cos(phi) * sin(theta) - dx;
    const double mis_y = part_length * sin(phi) - dy;
    const double mis_s = part_length * (cos(phi) * cos(theta) - 1) - ds;

    // Apply transformations
    START_PER_PARTICLE_BLOCK(part0, part);
        S_ROTATE(part, -psi);
        X_ROTATE(part, -phi);
        Y_ROTATE(part, -theta);
        S_SHIFT(part, mis_s);
        XY_SHIFT(part, mis_x, mis_y);
    END_PER_PARTICLE_BLOCK;
}


double sinc(double x) {
    if (fabs(x) < 1e-10) {
        return 1.0; // sinc(0) = 1
    } else {
        return sin(x) / x; // sinc(x) = sin(x)/x
    }
}


void matrix_multiply_4x4(const double a[4][4], const double b[4][4], double result[4][4]) {
    for (int i = 0; i < 4; i++) {
        result[i][0] = a[i][0] * b[0][0] + a[i][1] * b[1][0] + a[i][2] * b[2][0] + a[i][3] * b[3][0];
        result[i][1] = a[i][0] * b[0][1] + a[i][1] * b[1][1] + a[i][2] * b[2][1] + a[i][3] * b[3][1];
        result[i][2] = a[i][0] * b[0][2] + a[i][1] * b[1][2] + a[i][2] * b[2][2] + a[i][3] * b[3][2];
        result[i][3] = a[i][0] * b[0][3] + a[i][1] * b[1][3] + a[i][2] * b[2][3] + a[i][3] * b[3][3];
    }
}

#endif  // XTRACK_TRACK_MISALIGNMENT_H
