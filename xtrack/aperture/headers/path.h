#ifndef APERTURE_PATH_H
#define APERTURE_PATH_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "path.h"
#include "base.h"

/* 2D PATH made of segments

ISSUES:
 - no bezier segments yet
 - no splines yet
 - path assumed to be continuous but not enforced by the structure
*/

#ifndef M_2PI
    #define M_2PI (2.0 * M_PI)
#endif

#ifndef M_PI_2
    #define M_PI_2 (0.5 * M_PI)
#endif

#ifndef POINTS_PER_ARC
    #define POINTS_PER_ARC 10
#endif


typedef enum {
    LINE_SEGMENT_TYPE,
    ARC_SEGMENT_TYPE,
    ELLIPSE_ARC_SEGMENT_TYPE,
    // QUADRATIC_BEZIER_SEGMENT_TYPE,
    // CUBIC_BEZIER_SEGMENT_TYPE,
} SegmentType2D;


typedef struct {
    SegmentType2D type;
    union {
        struct { float_type x0, y0, x1, y1; } line_params;
        struct { float_type cx, cy, r, start_angle, end_angle; } arc_params;
        struct { float_type cx, cy, rx, ry, rotation, start_angle, end_angle; } ellipse_arc_params;
        // struct { float_type x1, y1, x2, y2, cx, cy; } quad_bezier_params;
        // struct { float_type x1, y1, x2, y2, cx1, cy1, cx2, cy2; } cubic_bezier_params;
    };
} Segment2D;


typedef struct {
    Segment2D *segments;
    int len_segments;
} Path2D;


/* Line segment functions */
void line_segment_from_start_end(float_type x0, float_type y0, float_type x1, float_type y1, Segment2D *out);
void line_segment_from_start_length(float_type x0, float_type y0, float_type dx, float_type dy, float_type length, Segment2D *out);
float_type line_segment_get_length(const Segment2D *seg);
void line_segment_get_points_at_steps(const Segment2D *seg, const float_type *steps, int len_points, Point2D *out_points);

/* Arc segment functions */
void arc_segment_from_center_radius_angles(float_type cx, float_type cy, float_type r, float_type start_angle, float_type end_angle, Segment2D *out);
void arc_segment_from_ref_length_angle(float_type x0, float_type y0, float_type dx, float_type dy, float_type length, float_type angle, Segment2D *out);
void arc_segment_get_ref_at_length(const Segment2D *seg, float_type at, float_type *out_x, float_type *out_y, float_type *out_dx, float_type *out_dy);
float_type arc_segment_get_length(const Segment2D *seg);
void arc_segment_get_points_at_steps(const Segment2D *seg, const float_type *steps, int len_points, Point2D *out_points);

/* Ellipse arc segment functions */
void ellipse_arc_segment_from_center_radii_rotation_angles(float_type cx, float_type cy, float_type rx, float_type ry, float_type rotation, float_type start_angle, float_type end_angle, Segment2D *out);
void maybe_ellipse_arc_segment_from_center_radii_rotation_angles(float_type cx, float_type cy, float_type rx, float_type ry, float_type rotation, float_type start_angle, float_type end_angle, Segment2D *out);
float_type ellipse_segment_get_length(const Segment2D *seg);
void ellipse_segment_get_points_at_steps(const Segment2D *seg, const float_type *steps, int len_points, Point2D *out_points);

/* Segment functions */
float_type segment2d_get_length(const Segment2D *seg);

/* Segments from shapes */
void segments_from_rectangle(float_type halfwidth, float_type halfheight, Segment2D *out_segments);
void segments_from_circle(float_type radius, Segment2D *out_segments);
void segments_from_ellipse(float_type rx, float_type ry, Segment2D *out_segments);
void segments_from_rectellipse(float_type halfwidth, float_type halfheight, float_type rx, float_type ry, Segment2D *out_segments, int *len_segments);
void segments_from_racetrack(float_type halfhside, float_type halfvside, float_type rx, float_type ry, Segment2D *out_segments, int *out_len);
void segments_from_octagon(float_type halfwidth, float_type halfheight, float_type halfdgap, Segment2D *out_segments, int *out_len);

/* Path functions */
int path_get_len_steps(const Path2D *path, float_type ds_min);
float_type path_get_length(const Path2D *path);
void path_get_steps(const Path2D *path, float_type ds_min, float_type *out_steps);
void path_get_points_at_steps(const Path2D *path, const float_type *steps, int len_points, Point2D *out_points);
int path_get_len_points(const Path2D *path);
void path_get_points(const Path2D *path, Point2D *out_points);
int path_get_len_corners(const Path2D *path);
void path_get_corner_steps(const Path2D *path, float_type *out_steps);
void poly_get_n_uniform_points(const Path2D *path, int n_points, Point2D *out_points);


/* ===== Line segment functions ===== */
void line_segment_from_start_end(float_type x0, float_type y0, float_type x1, float_type y1, Segment2D *out)
/* Get line data from starting and ending points

*/
{
    out->line_params.x0 = x0;
    out->line_params.y0 = y0;
    out->line_params.x1 = x1;
    out->line_params.y1 = y1;
    out->type = LINE_SEGMENT_TYPE;
}

void line_segment_from_start_length(float_type x0, float_type y0, float_type dx, float_type dy, float_type length, Segment2D *out)
/* Get line data from starting point, direction (assuming dx,dy have norm=1) and length

*/
{
    float_type ux = dx;
    float_type uy = dy;
    out->line_params.x0 = x0;
    out->line_params.y0 = y0;
    out->line_params.x1 = x0 + ux * length;
    out->line_params.y1 = y0 + uy * length;
    out->type = LINE_SEGMENT_TYPE;
}

float_type line_segment_get_length(const Segment2D *seg)
/* Get length of a line segment */
{
    float_type x1 = seg->line_params.x0;
    float_type y1 = seg->line_params.y0;
    float_type x2 = seg->line_params.x1;
    float_type y2 = seg->line_params.y1;
    return hypot(x2 - x1, y2 - y1);
}

void line_segment_get_points_at_steps(const Segment2D *seg, const float_type *steps, int len_points, Point2D *out_points)
/* Get points along a line segment at specified steps

Contract: len_points=len(steps); len(out_points)=len_points
*/
{
    float_type x1 = seg->line_params.x0;
    float_type y1 = seg->line_params.y0;
    float_type x2 = seg->line_params.x1;
    float_type y2 = seg->line_params.y1;
    float_type line_length = hypot(x2 - x1, y2 - y1);

    if (line_length <= APER_PRECISION)
    {
        for (int i = 0; i < len_points; i++)
        {
            out_points[i].x = x1;
            out_points[i].y = y1;
        }
        return;
    }

    float_type ux = (x2 - x1) / line_length;
    float_type uy = (y2 - y1) / line_length;

    for (int i = 0; i < len_points; i++)
    {
        float_type at = steps[i];
        out_points[i].x = x1 + ux * at;
        out_points[i].y = y1 + uy * at;
    }
}

/* ===== End line segment functions ===== */

/* ===== Arc segment functions ===== */

void arc_segment_from_center_radius_angles(float_type cx, float_type cy, float_type r, float_type start_angle, float_type end_angle, Segment2D *out)
/* Get arc data from center, radius and start/end angles

*/
{
    out->arc_params.cx = cx;
    out->arc_params.cy = cy;
    out->arc_params.r = r;
    out->arc_params.start_angle = start_angle;
    out->arc_params.end_angle = end_angle;
    out->type = ARC_SEGMENT_TYPE;
}

void arc_segment_from_ref_length_angle(float_type x0, float_type y0, float_type dx, float_type dy, float_type length, float_type angle, Segment2D *out)
/* Get arc data from starting point, direction (assuming dx,dy have norm=1), length and angle

*/
{
    float_type norm = hypot(dx, dy);
    if (angle == 0.0)
    {
        line_segment_from_start_length(x0, y0, dx, dy, length, out);
        return;
    }

    float_type r = length / fabs(angle);
    float_type cx = x0 - dy * r / norm;
    float_type cy = y0 + dx * r / norm;
    float_type start_angle = atan2(y0 - cy, x0 - cx);
    float_type end_angle = start_angle + angle;
    out->arc_params.cx = cx;
    out->arc_params.cy = cy;
    out->arc_params.r = r;
    out->arc_params.start_angle = start_angle;
    out->arc_params.end_angle = end_angle;
    out->type = ARC_SEGMENT_TYPE;
    return;
}

void arc_segment_get_ref_at_length(const Segment2D *seg, float_type at, float_type *out_x, float_type *out_y, float_type *out_dx, float_type *out_dy)
/* Get point and direction at length 'at' along an arc segment */
{
    /* arc */
    float_type cx = seg->arc_params.cx;
    float_type cy = seg->arc_params.cy;
    float_type r = seg->arc_params.r;
    float_type start_angle = seg->arc_params.start_angle;
    float_type end_angle = seg->arc_params.end_angle;
    float_type total_angle = end_angle - start_angle;
    float_type arc_length = fabs(total_angle) * r;

    if (arc_length <= APER_PRECISION)
    {
        *out_x = cx + r * cos(start_angle);
        *out_y = cy + r * sin(start_angle);
        *out_dx = -sin(start_angle);
        *out_dy = cos(start_angle);
        return;
    }

    float_type angle_at = start_angle + (at / arc_length) * total_angle;
    *out_x = cx + r * cos(angle_at);
    *out_y = cy + r * sin(angle_at);
    *out_dx = -sin(angle_at);
    *out_dy = cos(angle_at);
}

float_type arc_segment_get_length(const Segment2D *seg)
/* Get length of an arc segment */
{
    float_type r = seg->arc_params.r;
    float_type start_angle = seg->arc_params.start_angle;
    float_type end_angle = seg->arc_params.end_angle;
    float_type total_angle = end_angle - start_angle;
    return fabs(total_angle) * r;
}

void arc_segment_get_points_at_steps(const Segment2D *seg, const float_type *steps, int len_points, Point2D *out_points)
/* Get points along an arc segment at specified steps

Contract: len_points=len(steps); len(out_points)=len_points
*/
{
    float_type cx = seg->arc_params.cx;
    float_type cy = seg->arc_params.cy;
    float_type r = seg->arc_params.r;
    float_type start_angle = seg->arc_params.start_angle;
    float_type end_angle = seg->arc_params.end_angle;
    float_type total_angle = end_angle - start_angle;
    float_type arc_length = fabs(total_angle) * r;

    if (arc_length <= APER_PRECISION)
    {
        const float_type x0 = cx + r * cos(start_angle);
        const float_type y0 = cy + r * sin(start_angle);
        for (int i = 0; i < len_points; i++)
        {
            out_points[i].x = x0;
            out_points[i].y = y0;
        }
        return;
    }

    for (int i = 0; i < len_points; i++)
    {
        float_type at = steps[i];
        float_type angle_at = start_angle + (at / arc_length) * total_angle;
        out_points[i].x = cx + r * cos(angle_at);
        out_points[i].y = cy + r * sin(angle_at);
    }
}

/* ===== End arc segment functions ===== */

/* ===== Ellipse arc segment functions ===== */

void ellipse_arc_segment_from_center_radii_rotation_angles(float_type cx, float_type cy, float_type rx, float_type ry, float_type rotation, float_type start_angle, float_type end_angle, Segment2D *out)
{
    out->ellipse_arc_params.cx = cx;
    out->ellipse_arc_params.cy = cy;
    out->ellipse_arc_params.rx = rx;
    out->ellipse_arc_params.ry = ry;
    out->ellipse_arc_params.rotation = rotation;
    out->ellipse_arc_params.start_angle = start_angle;
    out->ellipse_arc_params.end_angle = end_angle;
    out->type = ELLIPSE_ARC_SEGMENT_TYPE;
}

void maybe_ellipse_arc_segment_from_center_radii_rotation_angles(float_type cx, float_type cy, float_type rx, float_type ry, float_type rotation, float_type start_angle, float_type end_angle, Segment2D *out)
{
    if (rx == ry)
    {
        float_type r = rx;
        arc_segment_from_center_radius_angles(cx, cy, r, start_angle, end_angle, out);
        return;
    }
    ellipse_arc_segment_from_center_radii_rotation_angles(cx, cy, rx, ry, rotation, start_angle, end_angle, out);
}

static float_type ellipse_cumulative_length(float_type angle, float_type rx, float_type ry)
/* Cumulative length of ellipse from 0 to angle */
{
    if (rx <= 0.0 || ry <= 0.0)
        return 0.0;
    if (rx == ry)
        return rx * fabs(angle);

    if (rx >= ry)
    {
        float_type k = sqrt(1.0 - (ry * ry) / (rx * rx));
        float_type complete_E = elliptic_E_complete(k);
        return rx * (complete_E - elliptic_E(0.5 * M_PI - angle, k));
    }

    float_type k = sqrt(1.0 - (rx * rx) / (ry * ry));
    return ry * elliptic_E(angle, k);
}

static float_type ellipse_arc_length_between(float_type start, float_type end, float_type rx, float_type ry)
/* Arc length of ellipse between start and end angles */
{
    return fabs(ellipse_cumulative_length(end, rx, ry) - ellipse_cumulative_length(start, rx, ry));
}

static float_type ellipse_angle_at_length(float_type start, float_type end, float_type rx, float_type ry, float_type target)
/* Find ellipse parameter angle at a given arc length from start towards end */
{
    if (rx <= 0.0 || ry <= 0.0)
        return start;
    float_type total_len = ellipse_arc_length_between(start, end, rx, ry);
    if (total_len == 0.0)
        return start;
    if (target <= 0.0)
        return start;
    if (target >= total_len)
        return end;

    float_type dir = (end >= start) ? 1.0 : -1.0;
    float_type theta_low = 0.0;
    float_type theta_high = fabs(end - start);

    if (rx == ry)
        return start + dir * (target / rx);

    for (int i = 0; i < 60; i++)
    {
        float_type theta_mid = 0.5 * (theta_low + theta_high);
        float_type angle_mid = start + dir * theta_mid;
        float_type len_mid = ellipse_arc_length_between(start, angle_mid, rx, ry);
        if (len_mid < target)
            theta_low = theta_mid;
        else
            theta_high = theta_mid;
    }
    float_type theta_mid = 0.5 * (theta_low + theta_high);
    return start + dir * theta_mid;
}

float_type ellipse_segment_get_length(const Segment2D *seg)
/* Get length of an ellipse arc segment */
{
    float_type rx = seg->ellipse_arc_params.rx;
    float_type ry = seg->ellipse_arc_params.ry;
    float_type start_angle = seg->ellipse_arc_params.start_angle;
    float_type end_angle = seg->ellipse_arc_params.end_angle;
    return ellipse_arc_length_between(start_angle, end_angle, rx, ry);
}

void ellipse_segment_get_points_at_steps(const Segment2D *seg, const float_type *steps, int len_points, Point2D *out_points)
/* Get points along an ellipse arc segment at specified steps

Contract: len(steps)=len_points; len(out_points)=len_points
*/
{
    float_type cx = seg->ellipse_arc_params.cx;
    float_type cy = seg->ellipse_arc_params.cy;
    float_type rx = seg->ellipse_arc_params.rx;
    float_type ry = seg->ellipse_arc_params.ry;
    float_type rotation = seg->ellipse_arc_params.rotation;
    float_type start_angle = seg->ellipse_arc_params.start_angle;
    float_type end_angle = seg->ellipse_arc_params.end_angle;

    float_type cos_rot = cos(rotation);
    float_type sin_rot = sin(rotation);

    for (int i = 0; i < len_points; i++)
    {
        float_type at = steps[i];
        float_type angle_at = ellipse_angle_at_length(start_angle, end_angle, rx, ry, at);
        float_type x_ellipse = rx * cos(angle_at);
        float_type y_ellipse = ry * sin(angle_at);
        // Apply rotation
        out_points[i].x = cx + (x_ellipse * cos_rot - y_ellipse * sin_rot);
        out_points[i].y = cy + (x_ellipse * sin_rot + y_ellipse * cos_rot);
    }
}

/* ===== End ellipse arc segment functions ===== */

/* ===== Racetrack segment functions ===== */
void segments_from_racetrack(float_type halfwidth, float_type halfheight, float_type rx, float_type ry, Segment2D *out_segments, int *out_len)
/* Create a path for a racetrack shape centered at (0,0)

Contract: len(out_segments)=8; postlen(out_segments)=out_len
*/
{
    if (rx <= 0.0 || ry <= 0.0)
    { // Just a rectangle
        segments_from_rectangle(halfwidth, halfheight, out_segments);
        *out_len = 4;
        return;
    }

    if (rx == halfwidth && ry == halfheight)
    {
        // Just an ellipse
        segments_from_ellipse(rx, ry, out_segments);
        *out_len = 1;
        return;
    }

    if (rx == halfwidth)
    {
        // Just vertical capsule flat side sides
        line_segment_from_start_end(halfwidth, -halfheight + ry, halfwidth, halfheight - ry, &out_segments[0]);
        maybe_ellipse_arc_segment_from_center_radii_rotation_angles(0, halfheight - ry, rx, ry, 0.0, 0, M_PI , &out_segments[1]);
        line_segment_from_start_end(-halfwidth, halfheight - ry, -halfwidth, -halfheight + ry, &out_segments[2]);
        maybe_ellipse_arc_segment_from_center_radii_rotation_angles(0, -halfheight + ry, rx, ry, 0.0, M_PI, 2.0 * M_PI, &out_segments[3]);
        *out_len = 4;
        return;
    }

    if (ry == halfheight)
    {
        // Just horizontal capsule flat top sides
        maybe_ellipse_arc_segment_from_center_radii_rotation_angles(halfwidth - rx, 0, rx, ry, 0.0, -M_PI / 2.0, M_PI / 2.0, &out_segments[0]);
        line_segment_from_start_end(halfwidth - rx, halfheight, -halfwidth + rx, halfheight, &out_segments[1]);
        maybe_ellipse_arc_segment_from_center_radii_rotation_angles(-halfwidth + rx, 0, rx, ry, 0.0, M_PI / 2.0, 3.0 * M_PI / 2.0, &out_segments[2]);
        line_segment_from_start_end(-halfwidth + rx, -halfheight, halfwidth - rx, -halfheight, &out_segments[3]);
        *out_len = 4;
        return;
    }

    // 8 corners
        line_segment_from_start_end(halfwidth, -halfheight + ry, halfwidth, halfheight - ry, &out_segments[0]);
        maybe_ellipse_arc_segment_from_center_radii_rotation_angles(halfwidth - rx, halfheight - ry, rx, ry, 0.0, 0, M_PI_2, &out_segments[1]);
        line_segment_from_start_end(halfwidth - rx, halfheight, -halfwidth + rx, halfheight, &out_segments[2]);
        maybe_ellipse_arc_segment_from_center_radii_rotation_angles(-halfwidth + rx, halfheight - ry, rx, ry, 0.0, M_PI_2, M_PI, &out_segments[3]);
        line_segment_from_start_end(-halfwidth, halfheight - ry, -halfwidth, -halfheight + ry, &out_segments[4]);
        maybe_ellipse_arc_segment_from_center_radii_rotation_angles(-halfwidth + rx, -halfheight + ry, rx, ry, 0.0, M_PI, 3.0 * M_PI_2, &out_segments[5]);
        line_segment_from_start_end(-halfwidth + rx, -halfheight, halfwidth - rx, -halfheight, &out_segments[6]);
        maybe_ellipse_arc_segment_from_center_radii_rotation_angles(halfwidth - rx, -halfheight + ry, rx, ry, 0.0, 3.0 * M_PI_2, M_2PI, &out_segments[7]);
        *out_len=8;

}
/* ===== End racetrack segment functions ===== */

void segments_from_octagon(float_type halfwidth, float_type halfheight, float_type halfdgap, Segment2D *out_segments, int *out_len)
/* Create an octagon defined by halfwidth, halfheight, and half 45 degrees gap

Contract: len(out_segments)=8; postlen(out_segments)=out_len
*/
{
    // Intersect of a line at 45 degrees with a minimum distance halfdgap from the cernter with the rectangle
    // y= halfdgap*sqrt(2)-x  or x = halfdgap*sqrt(2) - y
    float_type dx = halfdgap * sqrt(2.0) - halfheight;
    float_type dy = halfdgap * sqrt(2.0) - halfwidth;
    if (dx < 0.0 || dy < 0.0) {
        // Invalid parameters
        *out_len = 0;
        return;
    }

    line_segment_from_start_end(halfwidth, -dy, halfwidth, dy, &out_segments[0]);
    line_segment_from_start_end(halfwidth, dy, dx, halfheight, &out_segments[1]);
    line_segment_from_start_end(dx, halfheight, -dx, halfheight, &out_segments[2]);
    line_segment_from_start_end(-dx, halfheight, -halfwidth, dy, &out_segments[3]);
    line_segment_from_start_end(-halfwidth, dy, -halfwidth, -dy, &out_segments[4]);
    line_segment_from_start_end(-halfwidth, -dy, -dx, -halfheight, &out_segments[5]);
    line_segment_from_start_end(-dx, -halfheight, dx, -halfheight, &out_segments[6]);
    line_segment_from_start_end(dx, -halfheight, halfwidth, -dy, &out_segments[7]);
    *out_len = 8;

}

/* ===== End octagon segment functions ===== */

/* ===== Segment functions ===== */

float_type segment2d_get_length(const Segment2D *seg)
/* Get length of a segment */
{
    switch (seg->type)
    {
    case LINE_SEGMENT_TYPE:
        return line_segment_get_length(seg);
    case ARC_SEGMENT_TYPE:
        return arc_segment_get_length(seg);
    case ELLIPSE_ARC_SEGMENT_TYPE:
        return ellipse_segment_get_length(seg);
    default:
        return 0.0;
    }
}

/* ===== End segment functions ===== */

/* ===== Segments from shapes ===== */

void segments_from_rectangle(float_type halfwidth, float_type halfheight, Segment2D *out_segments)
/* Create a path for a rectangle centered at (0,0)

Contract: len(out_segments)=4
*/
{
    line_segment_from_start_end(-halfwidth, -halfheight, halfwidth, -halfheight, &out_segments[0]);
    line_segment_from_start_end(halfwidth, -halfheight, halfwidth, halfheight, &out_segments[1]);
    line_segment_from_start_end(halfwidth, halfheight, -halfwidth, halfheight, &out_segments[2]);
    line_segment_from_start_end(-halfwidth, halfheight, -halfwidth, -halfheight, &out_segments[3]);
}

void segments_from_circle(float_type r, Segment2D *out_segments)
/* Create a path for a circle centered at (0,0)

Contract: len(out_segments)=1
*/
{
    arc_segment_from_center_radius_angles(0.0, 0.0, r, 0.0, 2.0 * M_PI, &out_segments[0]);
}

void segments_from_ellipse(float_type rx, float_type ry, Segment2D *out_segments)
/* Create a path for an ellipse centered at (0,0)

Contract: len(out_segments)=1
*/
{
    if (rx == ry)
    {
        segments_from_circle(rx, out_segments);
        return;
    }
    ellipse_arc_segment_from_center_radii_rotation_angles(0.0, 0.0, rx, ry, 0.0, 0.0, 2.0 * M_PI, &out_segments[0]);
}

void segments_from_rectellipse(float_type halfwidth, float_type halfheight, float_type rx, float_type ry, Segment2D *out_segments, int *out_len)
/* Create a path for the intersection between a rectangle and an ellipse

Contract: len(out_segments)=8; postlen(out_segments)=out_len
*/
{
    float_type arg1 = halfwidth / rx;
    float_type arg2 = halfheight / ry;

    if (arg1 > 1.0)
        arg1 = 1.0;
    if (arg2 > 1.0)
        arg2 = 1.0;

    if (arg1 == 1.0 && arg2 == 1.0)
    {
        // printf("ellipse inside rectangle\n");
        segments_from_ellipse(rx, ry, out_segments);
        *out_len = 1;
        return;
    }

    float_type angle1 = acos(halfwidth / rx);
    float_type angle2 = asin(halfheight / ry);
    // printf("angle1=%f angle2=%f\n",angle1,angle2);

    if (angle2 < angle1)
    {
        // printf("rectangle inside ellipse\n");
        segments_from_rectangle(halfwidth, halfheight, out_segments);
        *out_len = 4;
        return;
    }

    float_type iy = ry * sin(angle1);
    float_type ix = rx * cos(angle2);

    if (arg1 < 1 && arg2 < 1)
    { // printf("8 segments\n");
        line_segment_from_start_end(halfwidth, -iy, halfwidth, iy, &out_segments[0]);
        maybe_ellipse_arc_segment_from_center_radii_rotation_angles(0.0, 0.0, rx, ry, 0.0, angle1, angle2, &out_segments[1]);
        line_segment_from_start_end(ix, halfheight, -ix, halfheight, &out_segments[2]);
        maybe_ellipse_arc_segment_from_center_radii_rotation_angles(0.0, 0.0, rx, ry, 0.0, M_PI - angle2, M_PI - angle1, &out_segments[3]);
        line_segment_from_start_end(-halfwidth, iy, -halfwidth, -iy, &out_segments[4]);
        maybe_ellipse_arc_segment_from_center_radii_rotation_angles(0.0, 0.0, rx, ry, 0.0, M_PI + angle1, M_PI + angle2, &out_segments[5]);
        line_segment_from_start_end(-ix, -halfheight, ix, -halfheight, &out_segments[6]);
        maybe_ellipse_arc_segment_from_center_radii_rotation_angles(0.0, 0.0, rx, ry, 0.0, M_2PI - angle2, M_2PI - angle1, &out_segments[7]);
        *out_len = 8;
        return;
    }
    if (arg1 < 1)
    { // printf("flat sides\n");
        line_segment_from_start_end(halfwidth, -iy, halfwidth, iy, &out_segments[0]);
        maybe_ellipse_arc_segment_from_center_radii_rotation_angles(0.0, 0.0, rx, ry, 0.0, angle1, M_PI - angle1, &out_segments[1]);
        line_segment_from_start_end(-halfwidth, iy, -halfwidth, -iy, &out_segments[2]);
        maybe_ellipse_arc_segment_from_center_radii_rotation_angles(0.0, 0.0, rx, ry, 0.0, M_PI + angle1, M_2PI - angle1, &out_segments[3]);
        *out_len = 4;
        return;
    }
    if (arg2 < 1)
    { // printf("flat top/bottom\n");
        maybe_ellipse_arc_segment_from_center_radii_rotation_angles(0.0, 0.0, rx, ry, 0.0, -angle2, angle2, &out_segments[0]);
        line_segment_from_start_end(ix, halfheight, -ix, halfheight, &out_segments[1]);
        maybe_ellipse_arc_segment_from_center_radii_rotation_angles(0.0, 0.0, rx, ry, 0.0, M_PI - angle2, M_PI + angle2, &out_segments[2]);
        line_segment_from_start_end(-ix, -halfheight, ix, -halfheight, &out_segments[3]);
        *out_len = 4;
        return;
    }
}

/* ===== End segments from shapes ===== */

/* ===== Path functions ===== */

int path_get_len_steps(const Path2D *path, float_type ds_min)
{
    /* Get the number of steps needed to represent a path defined by segments

    Contract: len_segments=len(segments)
    */
    int total_steps = 1;
    int nsteps;
    for (int i = 0; i < path->len_segments; i++)
    {
        switch (path->segments[i].type)
        {
        case 0: /* line */
            total_steps += ceil(line_segment_get_length(&path->segments[i]) / ds_min);
            break;
        case 1: /* arc */
            nsteps = ceil(arc_segment_get_length(&path->segments[i]) / ds_min);
            total_steps += nsteps > 10 ? nsteps : 10;
            break;
        case 2: /* ellipse arc */
            nsteps = ceil(ellipse_segment_get_length(&path->segments[i]) / ds_min);
            total_steps += nsteps > 10 ? nsteps : 10;
            break;
        default:
            break;
        }
    }
    return total_steps;
}

float_type path_get_length(const Path2D *path)
{
    /* Get length of a path defined by segments

    Contract: len_segments=len(segments)
    */
    float_type total_length = 0.0;
    for (int i = 0; i < path->len_segments; i++)
    {
        total_length += segment2d_get_length(&path->segments[i]);
    }
    return total_length;
}

void path_get_steps(const Path2D *path, float_type ds_min, float_type *out_steps)
{
    /* Get steps along a path defined by segments

    Contract: len_segments=len(segments); len(out_steps)=path_get_len_steps(segments,len_segments,ds_min)
    */
    int idx = 0;
    out_steps[idx++] = 0.0;
    float_type seg_length;
    float_type nsteps_d;
    int nsteps;
    float_type ds;
    float_type length_acc = 0.0;

    for (int i = 0; i < path->len_segments; i++)
    {
        float_type seg_start = length_acc;
        switch (path->segments[i].type)
        {
        case 0: /* line */
            seg_length = line_segment_get_length(&path->segments[i]);
            nsteps_d = seg_length / ds_min;
            nsteps = (int)ceil(nsteps_d);
            ds = seg_length / nsteps;
            for (int j = 1; j <= nsteps; j++)
            {
                if (j == nsteps)
                    length_acc = seg_start + seg_length;
                else
                    length_acc += ds;
                out_steps[idx++] = length_acc;
            }
            break;
        case 1: /* arc */
            seg_length = arc_segment_get_length(&path->segments[i]);
            nsteps_d = seg_length / ds_min;
            nsteps = (int)ceil(nsteps_d);
            if (nsteps < 10)
                nsteps = 10;
            ds = seg_length / nsteps;
            for (int j = 1; j <= nsteps; j++)
            {
                if (j == nsteps)
                    length_acc = seg_start + seg_length;
                else
                    length_acc += ds;
                out_steps[idx++] = length_acc;
            }
            break;
        case 2: /* ellipse arc */
            seg_length = ellipse_segment_get_length(&path->segments[i]);
            nsteps_d = seg_length / ds_min;
            nsteps = (int)ceil(nsteps_d);
            if (nsteps < 10)
                nsteps = 10;
            ds = seg_length / nsteps;
            for (int j = 1; j <= nsteps; j++)
            {
                if (j == nsteps)
                    length_acc = seg_start + seg_length;
                else
                    length_acc += ds;
                out_steps[idx++] = length_acc;
            }
            break;
        default:
            break;
        }
    }; // Ensure last step is exact length
}

void path_get_points_at_steps(const Path2D *path, const float_type *steps, int len_points, Point2D *out_points)
/* Get points along a path defined by segments at specified steps

Contract: len_points=len(steps); len(out_points)=len_points
*/
{
    int i = 0;
    float_type seg_start_length = 0.0;
    float_type seg_end_length = 0.0;
    float_type seg_length;

    for (int seg_idx = 0; seg_idx < path->len_segments; seg_idx++)
    {
        const Segment2D* segment = &path->segments[seg_idx];
        seg_length = segment2d_get_length(segment);
        seg_end_length = seg_start_length + seg_length;

        while (i < len_points && steps[i] <= seg_end_length)
        {
            float_type at = steps[i] - seg_start_length;
            switch (path->segments[seg_idx].type)
            {
            case 0: /* line */
                line_segment_get_points_at_steps(segment, &at, 1, &out_points[i]);
                break;
            case 1: /* arc */
                arc_segment_get_points_at_steps(segment, &at, 1, &out_points[i]);
                break;
            case 2: /* ellipse arc */
                ellipse_segment_get_points_at_steps(segment, &at, 1, &out_points[i]);
                break;
            default:
                break;
            }
            i++;
        }
        seg_start_length = seg_end_length;
    }
}

int path_get_len_points(const Path2D *path)
/*Return the number of points needed for a good representation of the path

*/
{
    int total_points = 1;
    for (int i = 0; i < path->len_segments; i++)
    {
        if (path->segments[i].type == 0)
        {
            total_points += 1;
        }
        else if (path->segments[i].type == 1 || path->segments[i].type == 2)
        {
            total_points += POINTS_PER_ARC;
        }
    }
    return total_points;
}

void path_get_points(const Path2D *path, Point2D *out_points)
/* Get points along a path defined by segments for a good representation of the path

Contract: len(out_points)=path_get_len_points(path)
*/
{
    int idx = 0;
    const float_type steps[] = {0.0};
    switch (path->segments[0].type) {
        case 0: /* line segment */
            line_segment_get_points_at_steps(&path->segments[0], steps, 1, &out_points[idx]);
            break;
        case 1: /* arc segment */
            arc_segment_get_points_at_steps(&path->segments[0], steps, 1, &out_points[idx]);
            break;
        case 2: /* ellipse arc */
            ellipse_segment_get_points_at_steps(&path->segments[0], steps, 1, &out_points[idx]);
            break;
    }
    idx++;

    for (int i = 0; i < path->len_segments; i++)
    {
        if (path->segments[i].type == 0)
        {
            const float_type seg_length = line_segment_get_length(&path->segments[i]);
            const float_type steps[] = {seg_length};
            line_segment_get_points_at_steps(&path->segments[i], steps, 1, &out_points[idx]);
            idx += 1;
        }
        else if (path->segments[i].type == 1)
        {
            float_type seg_length = arc_segment_get_length(&path->segments[i]);
            float_type ds = seg_length / POINTS_PER_ARC;
            float_type steps[POINTS_PER_ARC];
            for (int j = 1; j <= POINTS_PER_ARC; j++)
            {
                steps[j - 1] = j * ds;
            }
            arc_segment_get_points_at_steps(&path->segments[i], steps, POINTS_PER_ARC, &out_points[idx]);
            idx += POINTS_PER_ARC;
        }
        else if (path->segments[i].type == 2)
        {
            float_type seg_length = ellipse_segment_get_length(&path->segments[i]);
            float_type ds = seg_length / POINTS_PER_ARC;
            float_type steps[POINTS_PER_ARC];
            for (int j = 1; j <= POINTS_PER_ARC; j++)
            {
                steps[j - 1] = j * ds;
            }
            ellipse_segment_get_points_at_steps(&path->segments[i], steps, POINTS_PER_ARC, &out_points[idx]);
            idx += POINTS_PER_ARC;
        }
    }
}

int path_get_len_corners(const Path2D *path)
{
    /*Return number of corners in path

    */
    return path->len_segments + 1;
}

void path_get_corner_steps(const Path2D *path, float_type *out_steps)
{
    /*Get steps at corners of path

    Contract: len(out_steps)=path_get_len_corners(path)
    */
    float_type length_acc = 0.0;
    out_steps[0] = 0.0;
    for (int i = 0; i < path->len_segments; i++)
    {
        length_acc += segment2d_get_length(&path->segments[i]);
        out_steps[i + 1] = length_acc;
    }
}

void poly_get_n_uniform_points(const Path2D *path, int n_points, Point2D *out_points)
{
    float_type total_length = path_get_length(path);
    float_type ds = total_length / (n_points - 1);

    int i = 0;
    float_type seg_start_length = 0.0;
    float_type seg_end_length = 0.0;
    float_type seg_length;

    for (int seg_idx = 0; seg_idx < path->len_segments; seg_idx++)
    {
        const Segment2D* segment = &path->segments[seg_idx];
        seg_length = segment2d_get_length(segment);
        seg_end_length = seg_start_length + seg_length;
        float_type at_absolute;

        while (i < n_points - 1 && (at_absolute = i * ds) <= seg_end_length)
        {
            float_type at = at_absolute - seg_start_length;
            switch (path->segments[seg_idx].type)
            {
            case LINE_SEGMENT_TYPE:
                line_segment_get_points_at_steps(segment, &at, 1, &out_points[i]);
                break;
            case ARC_SEGMENT_TYPE:
                arc_segment_get_points_at_steps(segment, &at, 1, &out_points[i]);
                break;
            case ELLIPSE_ARC_SEGMENT_TYPE:
                ellipse_segment_get_points_at_steps(segment, &at, 1, &out_points[i]);
                break;
            default:
                break;
            }
            i++;
        }
        seg_start_length = seg_end_length;
    }

    // Close the polygon explicitly: avoids numerical issues on computing the last point, such as overshooting.
    // Also, if the start and end points are slightly different it can knock off the point-in-poly test.
    out_points[n_points - 1] = out_points[0];
}

#endif /* APERTURE_PATH_H */
