// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_HENONMAP_H
#define XTRACK_HENONMAP_H


/*gpufun*/
void Henonmap_track_local_particle(HenonmapData el, LocalParticle* part0){

    double const sin_omega_x = HenonmapData_get_sin_omega_x(el);
    double const cos_omega_x = HenonmapData_get_cos_omega_x(el);
    double const sin_omega_y = HenonmapData_get_sin_omega_y(el);
    double const cos_omega_y = HenonmapData_get_cos_omega_y(el);

    int const n_turns = HenonmapData_get_n_turns(el);

    int const n_fx_coeffs = HenonmapData_get_n_fx_coeffs(el);
    int const n_fy_coeffs = HenonmapData_get_n_fy_coeffs(el);

    double const alpha_x = HenonmapData_get_twiss_params(el, 0);
    double const beta_x = HenonmapData_get_twiss_params(el, 1);
    double const alpha_y = HenonmapData_get_twiss_params(el, 2);
    double const beta_y = HenonmapData_get_twiss_params(el, 3);
    double const sqrt_beta_x = sqrt(beta_x);
    double const sqrt_beta_y = sqrt(beta_y);

    int const norm = HenonmapData_get_norm(el);

    double fx_coeffs[128];
    int fx_x_exps[128];
    int fx_y_exps[128];
    for (int i = 0; i < n_fx_coeffs; i++)
    {
        fx_coeffs[i] = HenonmapData_get_fx_coeffs(el, i);
        fx_x_exps[i] = HenonmapData_get_fx_x_exps(el, i);
        fx_y_exps[i] = HenonmapData_get_fx_y_exps(el, i);
    }

    double fy_coeffs[128];
    int fy_x_exps[128];
    int fy_y_exps[128];
    for (int i = 0; i < n_fy_coeffs; i++)
    {
        fy_coeffs[i] = HenonmapData_get_fy_coeffs(el, i);
        fy_x_exps[i] = HenonmapData_get_fy_x_exps(el, i);
        fy_y_exps[i] = HenonmapData_get_fy_y_exps(el, i);
    }

    //start_per_particle_block (part0->part)

        double x = LocalParticle_get_x(part);
        double px = LocalParticle_get_px(part);
        double y = LocalParticle_get_y(part);
        double py = LocalParticle_get_py(part);

        double x_hat, px_hat, y_hat, py_hat;
        if (norm)
        {
            x_hat = x;
            px_hat = px;
            y_hat = y;
            py_hat = py;
        }
        else
        {
            x_hat = x / sqrt_beta_x;
            px_hat = alpha_x * x / sqrt_beta_x + px * sqrt_beta_x;
            y_hat = y / sqrt_beta_y;
            py_hat = alpha_y * y / sqrt_beta_y + py * sqrt_beta_y;
        }

        for (int n = 0; n < n_turns; n++)
        {

            double fx = 0;
            for (int i = 0; i < n_fx_coeffs; i++)
            {
                double prod = fx_coeffs[i];
                int x_power = fx_x_exps[i];
                int y_power = fx_y_exps[i];
                for (int j = 0; j < x_power; j++)
                {
                    prod *= (sqrt_beta_x * x_hat);
                }
                for (int j = 0; j < y_power; j++)
                {
                    prod *= (sqrt_beta_y * y_hat);
                }
                fx += prod;
            }
            double fy = 0;
            for (int i = 0; i < n_fy_coeffs; i++)
            {
                double prod = fy_coeffs[i];
                int x_power = fy_x_exps[i];
                int y_power = fy_y_exps[i];
                for (int j = 0; j < x_power; j++)
                {
                    prod *= (sqrt_beta_x * x_hat);
                }
                for (int j = 0; j < y_power; j++)
                {
                    prod *= (sqrt_beta_y * y_hat);
                }
                fy += prod;
            }
            fx *= sqrt_beta_x;
            fy *= sqrt_beta_y;

            double x_hat_new, px_hat_new, y_hat_new, py_hat_new;
            x_hat_new = cos_omega_x * x_hat + sin_omega_x * (px_hat + fx);
            px_hat_new = -sin_omega_x * x_hat + cos_omega_x * (px_hat + fx);
            y_hat_new = cos_omega_y * y_hat + sin_omega_y * (py_hat + fy);
            py_hat_new = -sin_omega_y * y_hat + cos_omega_y * (py_hat + fy);
            x_hat = x_hat_new;
            px_hat = px_hat_new;
            y_hat = y_hat_new;
            py_hat = py_hat_new;

        }

        if (norm)
        {
            x = x_hat;
            px = px_hat;
            y = y_hat;
            py = py_hat;
        }
        else
        {
            x = sqrt_beta_x * x_hat;
            px = -alpha_x * x_hat / sqrt_beta_x + px_hat / sqrt_beta_x;
            y = sqrt_beta_y * y_hat;
            py = -alpha_y * y_hat / sqrt_beta_y + py_hat / sqrt_beta_y;
        }

        LocalParticle_set_x(part, x);
        LocalParticle_set_px(part, px);
        LocalParticle_set_y(part, y);
        LocalParticle_set_py(part, py);

    //end_per_particle_block

}


#endif /* XTRACK_HENONMAP_H */
