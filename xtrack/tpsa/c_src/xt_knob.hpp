/* Address-keyed knob table (tpsa_param flavor only).
 *
 * Maps a strength double's buffer address to a parametric TPSA. The generated element
 * getters call xt_knob() at the moment of the read, so a registered address enters the
 * physics as a TPSA in the knob parameters and every other field as a constant TPSA.
 * The key is computed from the live el pointer, so there is no staleness window.
 * The table is set from Python before every track via the per-flavor setter below.
 *
 * Compiled only under XT_KNOBS (mad::tpsa is XT_STRENGTH there).
 */
#ifndef XT_KNOB_HPP
#define XT_KNOB_HPP

/* mad::tpsa is non-movable (TPSA_USE_TMP). Value-producing expressions return this
 * temporary type, which the generated getters hand back by value. */
typedef mad::mad_prv_::tpsa_tmp_ xt_knob_tpsa;

#define XT_KNOB_MAX 1024
static const double* xt_knob_addr[XT_KNOB_MAX];
static const tpsa_t* xt_knob_val [XT_KNOB_MAX];
static const tpsa_t* xt_knob_proto = 0;   /* descriptor carrier used for TPSA creation */
/* how many entries of xt_knob_addr and xt_knob_val are live (number of affected elements by knobs) */
static int64_t       xt_knob_n = 0;


/* Set from Python before every track. */
extern "C" void XT_F(xt_bridge_set_knob_table)(
        const void** addrs, const void** tpsas, const void* proto, int64_t n){
    if (n > XT_KNOB_MAX) n = 0;   /* refuse to overflow: treat as "no knobs" */
    for (int64_t i = 0; i < n; ++i){
        xt_knob_addr[i] = (const double*) addrs[i];
        xt_knob_val [i] = (const tpsa_t*) tpsas[i];
    }
    xt_knob_proto = (const tpsa_t*) proto;
    xt_knob_n = n;
}

/* A double as a constant TPSA on the prototype descriptor. */
static inline xt_knob_tpsa xt_knob_lift(double v){
    return 0.0 * mad::tpsa_ref((tpsa_t*) xt_knob_proto) + v;
}

/* If addr is a registered knob, return a value copy of its parametric TPSA
 * (`1.0 * ref` = scl -> a fresh temporary carrying the param slots). Otherwise return
 * `raw` as a constant TPSA on the prototype (coordinate) descriptor. Both branches
 * yield the temporary type, so the result is movable/returnable. */
static inline xt_knob_tpsa xt_knob(const double* addr, double raw){
    for (int64_t i = 0; i < xt_knob_n; ++i)
        if (xt_knob_addr[i] == addr)
            return 1.0 * mad::tpsa_ref((tpsa_t*) xt_knob_val[i]);
    return 0.0 * mad::tpsa_ref((tpsa_t*) xt_knob_proto) + raw;
}
#endif /* XT_KNOB_HPP */
