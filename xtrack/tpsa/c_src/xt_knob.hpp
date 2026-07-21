/* Address-keyed knob table (Route B, tpsa_param flavor only).
 *
 * Maps a strength double's buffer address to a parametric TPSA. A knob-using magnet
 * header records its strength field addresses (XT_KNOB_SET) before calling
 * track_magnet_particles, whose internal lift (XT_K) looks them up here.
 * A registered address enters the physics as a TPSA in the knob parameters,
 * every other field as a constant TPSA. The table is (re)set from Python before every
 * track (buffer-realloc danger) via the per-flavor setter below.
 *
 * Compiled only under XT_KNOBS (mad::tpsa is XT_STRENGTH there).
 */
#ifndef XT_KNOB_HPP
#define XT_KNOB_HPP

/* mad::tpsa is non-movable (TPSA_USE_TMP). Value-producing expressions return this
 * temporary type, which the XT_K lift hands back by value. */
typedef mad::mad_prv_::tpsa_tmp_ xt_knob_tpsa;

#define XT_KNOB_MAX 1024
static const double* xt_knob_addr[XT_KNOB_MAX];
static const tpsa_t* xt_knob_val [XT_KNOB_MAX];
static const tpsa_t* xt_knob_proto = 0;   /* descriptor carrier (a coordinate handle) */
static int64_t       xt_knob_n = 0;

/* Per-element strength-address slots (0..7 = k0,k1,k2,k3,k0s,k1s,k2s,k3s). A magnet
 * header affected by knobs fills the slots it uses (XT_KNOB_SET). XT_KNOB_CLEAR
 * clears them before each element so a non-setting element never reads stale slots. */
#define XT_KNOB_NSLOT 8
static const double* xt_cur_addr[XT_KNOB_NSLOT];
static inline void xt_knob_clear_current(){
    for (int i = 0; i < XT_KNOB_NSLOT; ++i) xt_cur_addr[i] = 0;
}

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
