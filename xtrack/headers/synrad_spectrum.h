// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_SYNRAD_SPECTRUM_H
#define XTRACK_SYNRAD_SPECTRUM_H

#define SQRT3 1.732050807568877
#define ALPHA_EM 0.0072973525693

/*gpufun*/
void synrad_average_kick(LocalParticle* part, double curv, double lpath,
                         double* dp_record, double* dpx_record, double* dpy_record
                        ){
    double const gamma0  = LocalParticle_get_gamma0(part);
    double const beta0  = LocalParticle_get_beta0(part);
    double const mass0 = LocalParticle_get_mass0(part);
    double const q0 = LocalParticle_get_q0(part);

    double const delta  = LocalParticle_get_delta(part);

    double const r = QELEM/(6*PI*EPSILON_0)
                        * q0*q0 / mass0
                        * (beta0*gamma0)*(beta0*gamma0)*(beta0*gamma0)
	                * curv*curv
                        * lpath * (1 + delta);

    double const beta = beta0 * LocalParticle_get_rvv(part);
    double f_t = sqrt(1 + r*(r-2)/(beta*beta));

    #ifdef XTRACK_SYNRAD_SCALE_SAME_AS_FIRST
    if (part -> ipart == 0){
      *dp_record = f_t;
    }
    else{
      f_t = *dp_record;
    }
    #endif

    #ifdef XTRACK_SYNRAD_KICK_SAME_AS_FIRST
    if (part -> ipart == 0){
      *dp_record = LocalParticle_get_delta(part);
      *dpx_record = LocalParticle_get_px(part);
      *dpy_record = LocalParticle_get_py(part);
    }
    else{
      f_t = 1.0;
    }
    #endif

    LocalParticle_update_delta(part, (delta+1) * f_t - 1);
    LocalParticle_scale_px(part, f_t);
    LocalParticle_scale_py(part, f_t);

    #ifdef XTRACK_SYNRAD_KICK_SAME_AS_FIRST
    if (part -> ipart == 0){
      *dp_record = LocalParticle_get_delta(part) - *dp_record;
      *dpx_record = LocalParticle_get_px(part) - *dpx_record;
      *dpy_record = LocalParticle_get_py(part) - *dpy_record;
    }
    else{
      LocalParticle_update_delta(part, LocalParticle_get_delta(part) + *dp_record);
      LocalParticle_add_to_px(part, *dpx_record);
      LocalParticle_add_to_py(part, *dpy_record);
    }
    #endif
}

/*gpufun*/
double SynRad(double x)
{
  // x :    energy normalized to the critical energy
  // returns function value _SynRadC   photon spectrum dn/dx
  // (integral of modified 1/3 order Bessel function)
  // principal: Chebyshev series see H.H.Umstaetter CERN/PS/SM/81-13 10-3-1981
  // see also my LEP Note 632 of 12/1990
  // converted to C++, H.Burkhardt 21-4-1996    */
  double synrad = 0.;
  if(x>0. && x<800.) {	// otherwise result synrad remains 0
    if(x<6.) {
      double a,b,z;
      z=x*x/16.-2.;
      b=          .00000000000000000012;
      a=z*b  +    .00000000000000000460;
      b=z*a-b+    .00000000000000031738;
      a=z*b-a+    .00000000000002004426;
      b=z*a-b+    .00000000000111455474;
      a=z*b-a+    .00000000005407460944;
      b=z*a-b+    .00000000226722011790;
      a=z*b-a+    .00000008125130371644;
      b=z*a-b+    .00000245751373955212;
      a=z*b-a+    .00006181256113829740;
      b=z*a-b+    .00127066381953661690;
      a=z*b-a+    .02091216799114667278;
      b=z*a-b+    .26880346058164526514;
      a=z*b-a+   2.61902183794862213818;
      b=z*a-b+  18.65250896865416256398;
      a=z*b-a+  92.95232665922707542088;
      b=z*a-b+ 308.15919413131586030542;
      a=z*b-a+ 644.86979658236221700714;
      double p;
      p=.5*z*a-b+  414.56543648832546975110;
      a=          .00000000000000000004;
      b=z*a+      .00000000000000000289;
      a=z*b-a+    .00000000000000019786;
      b=z*a-b+    .00000000000001196168;
      a=z*b-a+    .00000000000063427729;
      b=z*a-b+    .00000000002923635681;
      a=z*b-a+    .00000000115951672806;
      b=z*a-b+    .00000003910314748244;
      a=z*b-a+    .00000110599584794379;
      b=z*a-b+    .00002581451439721298;
      a=z*b-a+    .00048768692916240683;
      b=z*a-b+    .00728456195503504923;
      a=z*b-a+    .08357935463720537773;
      b=z*a-b+    .71031361199218887514;
      a=z*b-a+   4.26780261265492264837;
      b=z*a-b+  17.05540785795221885751;
      a=z*b-a+  41.83903486779678800040;
      double q;
      q=.5*z*a-b+28.41787374362784178164;
      double y;
      y=pow(x,2./3.);
      synrad=(p/y-q*y-1.)*1.81379936423421784215530788143;

    } else {// 6 < x < 174

      double a,b,z;
      z=20./x-2.;
      a=      .00000000000000000001;
      b=z*a  -.00000000000000000002;
      a=z*b-a+.00000000000000000006;
      b=z*a-b-.00000000000000000020;
      a=z*b-a+.00000000000000000066;
      b=z*a-b-.00000000000000000216;
      a=z*b-a+.00000000000000000721;
      b=z*a-b-.00000000000000002443;
      a=z*b-a+.00000000000000008441;
      b=z*a-b-.00000000000000029752;
      a=z*b-a+.00000000000000107116;
      b=z*a-b-.00000000000000394564;
      a=z*b-a+.00000000000001489474;
      b=z*a-b-.00000000000005773537;
      a=z*b-a+.00000000000023030657;
      b=z*a-b-.00000000000094784973;
      a=z*b-a+.00000000000403683207;
      b=z*a-b-.00000000001785432348;
      a=z*b-a+.00000000008235329314;
      b=z*a-b-.00000000039817923621;
      a=z*b-a+.00000000203088939238;
      b=z*a-b-.00000001101482369622;
      a=z*b-a+.00000006418902302372;
      b=z*a-b-.00000040756144386809;
      a=z*b-a+.00000287536465397527;
      b=z*a-b-.00002321251614543524;
      a=z*b-a+.00022505317277986004;
      b=z*a-b-.00287636803664026799;
      a=z*b-a+.06239591359332750793;
      double p;
      p=.5*z*a-b    +1.06552390798340693166;
      synrad=p*sqrt(0.5*PI/x)/exp(x);
    }
  }
  return synrad;
}

/*gpufun*/
double synrad_gen_photon_energy_normalized(LocalParticle *part)
{
  // initialize constants used in the approximate expressions
  // for SYNRAD   (integral over the modified Bessel function K5/3)
  //  xmin = 0.;
  double const xlow = 1.;
  double const a1 = 2.149528241534391; // Synrad(1.e-38)/pow(1.e-38,-2./3.);
  double const a2 = 1.770750801624037; // Synrad(xlow)/exp(-xlow);
  double const c1 = 0.; //
  double const ratio = 0.908250405131381;
  double appr, exact, result;
  do {
    if (RandomUniform_generate(part) < ratio) { // use low energy approximation
      result=c1+(1.-c1)*RandomUniform_generate(part);
      double tmp = result*result;
      result*=tmp;  	// take to 3rd power;
      exact=SynRad(result);
      appr=a1/tmp;
    } else {				// use high energy approximation
      result=xlow-log(RandomUniform_generate(part));
      exact=SynRad(result);
      appr=a2*exp(-result);
    }
  } while (exact < appr*RandomUniform_generate(part));	// reject in proportion of approx
  return result; // result now exact spectrum with unity weight
}

/*gpufun*/
double synrad_average_number_of_photons(
                          double beta0_gamma0, double curv, double lpath){
    double const kick = curv * lpath;
    return 2.5/SQRT3*ALPHA_EM*beta0_gamma0*fabs(kick);
}

/*gpufun*/
int64_t synrad_emit_photons(LocalParticle *part, double curv /* 1/m */,
                            double lpath /* m */,
                            RecordIndex record_index,
                            SynchrotronRadiationRecordData record
                            ){

    if (fabs(curv) < 1e-15)
        return 0;

    int64_t nphot = 0;

    // TODO Introduce effect of chi and mass_ratio!!!
    double const m0 = LocalParticle_get_mass0(part); // eV
    double const gamma0  = LocalParticle_get_gamma0(part);
    double const beta0  = LocalParticle_get_beta0(part);

    double const initial_energy = LocalParticle_get_energy0(part)
	                          + LocalParticle_get_ptau(part)*LocalParticle_get_p0c(part); // eV
    double energy = initial_energy;
    double gamma = energy / m0; //
    //double beta_gamma = sqrt(gamma*gamma-1); //
    double n = RandomExponential_generate(part); // path_length / mean_free_path;
    while (n < synrad_average_number_of_photons(beta0 * gamma0, curv, lpath)) {
        nphot++;
        double const c1 = 1.5 * 1.973269804593025e-07; // hbar * c = 1.973269804593025e-07 eV * m
        double const energy_critical = c1 * (gamma*gamma*gamma0) * curv; // eV
        double const energy_loss = synrad_gen_photon_energy_normalized(part) * energy_critical; // eV
        if (energy_loss >= energy) {
            energy = 0.0; // eV
            break;
        }
        energy -= energy_loss; // eV
        gamma = energy / m0; //
        // beta_gamma = sqrt(gamma*gamma-1); // that's how beta gamma is
        n += RandomExponential_generate(part);
        if (record){
          int64_t i_slot = RecordIndex_get_slot(record_index);
          // The returned slot id is negative if record is NULL or if record is full

          if (i_slot>=0){
              SynchrotronRadiationRecordData_set_photon_energy(record, i_slot,
                                                               energy_loss);
              SynchrotronRadiationRecordData_set_at_element(record, i_slot,
                                          LocalParticle_get_at_element(part));
              SynchrotronRadiationRecordData_set_at_turn(record, i_slot,
                                          LocalParticle_get_at_turn(part));
              SynchrotronRadiationRecordData_set_particle_id(record, i_slot,
                                          LocalParticle_get_particle_id(part));
              SynchrotronRadiationRecordData_set_particle_delta(record, i_slot,
                                          LocalParticle_get_delta(part));
          }
        }
    }

    if (energy == 0.0)
      LocalParticle_set_state(part, XT_LOST_ALL_E_IN_SYNRAD); // used to flag this kind of loss
    else{
      LocalParticle_add_to_energy(part, energy-initial_energy, 0);
    }

    return nphot;
}

#endif /* XTRACK_SYNRAD_SPECTRUM_H */
