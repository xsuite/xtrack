/* Single-particle LocalParticle for running Xsuite element physics (with generic XT_NUM type) on
 * either plain doubles or TPSA. The flavor is decided at compile time: -DXT_FLAVOR_TPSA
 * (coords are tpsa_t*) or -DXT_FLAVOR_NUM (coords are double*). This is included before the
 * element headers so track.h's START_PER_PARTICLE_BLOCK + the physics see this
 * LocalParticle and XT_NUM. Everything field-related (get/set/add/scale accessors) is
 * generated into xt_local_particle_gen.hpp. Here, only the struct and the the fixed
 * glue that a Taylor map needs are kept (loss/bookkeeping stubs). */
#ifndef XT_LOCAL_PARTICLE_HPP
#define XT_LOCAL_PARTICLE_HPP

#include <cstdint>

/* XS_FLAG_* bit positions, generated from xtrack's track_flags.py. The bridge
 * always runs with track_flags==0, so every flag reads false (no backtrack, no taper,
 * cavity kick applied, local aperture checked). */
#include "generated/xt_track_flags.h"

#if defined(XT_FLAVOR_TPSA)
  #include "mad_tpsa.hpp"      /* mad::tpsa / tpsa_ref, operators, sqrt */
  /* Not `using namespace mad;` because it pulls mad's scalar helpers
   * like (e.g. mad::fabs(num_t)) into the global overload set, making the physics'
   * fabs(double) ambiguous with ::fabs(double). Tpsa-typed math calls (sqrt/sin/fabs
   * on coordinates) resolve via argument-dependent lookup since the tpsa types live
   * in namespace mad. Only double-typed calls need the global/std versions, so ADL alone is correct here. */
  using mad::tpsa;
  using mad::tpsa_ref;
  #define XT_NUM   mad::tpsa
  typedef tpsa_t XT_COORD;
  /* pass coordinates by const-reference: a tpsa by-value copy only copies descriptor-only, not the coefficients (value lost) */
  #define XT_NUM_CONST_ARG const XT_NUM&

  /* physics branches on coordinates (if(Kx>0), NONZERO(K), ...) evaluate on the const
   * part (order 0), matching MAD-NG damap tracking semantics. mad_tpsa.hpp defines no
   * relational operators, so add them here (tpsa vs double, both directions, and tpsa vs
   * tpsa). a[0] == mad_tpsa_geti(a.ptr(),0) is the const part. */
  #define XT_TPSA_REL(OP) \
    template<class A> inline bool operator OP (const mad::tpsa_base<A>& a, double b){ return a[0] OP b; } \
    template<class A> inline bool operator OP (double a, const mad::tpsa_base<A>& b){ return a OP b[0]; } \
    template<class A, class B> inline bool operator OP (const mad::tpsa_base<A>& a, const mad::tpsa_base<B>& b){ return a[0] OP b[0]; }
  XT_TPSA_REL(>) XT_TPSA_REL(<) XT_TPSA_REL(>=) XT_TPSA_REL(<=) XT_TPSA_REL(==) XT_TPSA_REL(!=)
  #undef XT_TPSA_REL

  /* Parametric-knob build: lattice strengths become TPSAs so knob dependence is in
   * the maps. Plain tpsa flavor keeps XT_STRENGTH == double (track.h default). */
  #if defined(XT_KNOBS)
    #include <vector>           /* lifting double multipole arrays -> constant tpsa */
    #define XT_STRENGTH mad::tpsa
    #define XT_STRENGTH_CONST_ARG const XT_STRENGTH&
    /* A tpsa cannot be a mutable by-value parameter, so strengths arrive by const
     * reference and tapering is unavailable here. */
    #define XT_STRENGTH_ARG const XT_STRENGTH&
    /* Const part, for the double-only edge path. */
    #define XT_STRENGTH_CONST(v) ((v)[0])
    /* Lift a double to a constant tpsa. Defined in xt_knob.hpp, which owns the
     * prototype descriptor the lift borrows. */
    #define XT_STRENGTH_LIFT(v) xt_knob_lift(v)
    /* fabs(tpsa) = |const part| is already provided by mad_tpsa.hpp and found via argument-dependent lookup
     * (the tpsa types live in namespace mad). This matches the XT_TPSA_REL const-part branching. */
  #endif
#elif defined(XT_FLAVOR_NUM)
  #include <math.h>           /* global sqrt(double) for the physics' unqualified sqrt */
  #define XT_NUM   double
  typedef double XT_COORD;
#else
  #error "define XT_FLAVOR_TPSA or XT_FLAVOR_NUM"
#endif

/* The bridge ABI struct is an xobject (xtrack.tpsa._bridge_particle.XtBridgeParticle).
 * Its C type is the opaque `struct XtBridgeParticle_s *` typedef `XtBridgeParticle`,
 * defined (with its accessors) by the generated element C-API. We only ever use it
 * through those accessors, so an incomplete forward declaration is enough here. */
struct XtBridgeParticle_s;

/* Coords/derived are pointers so the map is read/written in place. Reference scalars
 * and the int bookkeeping are read through `bp` (bridge particle) via the generated XtBridgeParticle_get_*
 * accessors (bp is an opaque xobject pointer, not a plain struct). The struct's field
 * set is emitted from xtrack's own struct generator (same var lists as the native SoA
 * LocalParticle) -- included here after the XT_COORD #define + the XtBridgeParticle_s
 * forward decl it references. */
#include "generated/xt_local_particle_struct.h"

/* The generated field accessors (xt_local_particle_gen.hpp) are included by
 * xt_bridge.cpp after generated/xt_bridge_particle.h, since they dereference the
 * complete XtBridgeParticle through `bp`. */

/* Fixed glue a single-particle map needs but which is not field-shaped. */
static inline int64_t LocalParticle_get__num_active_particles(LocalParticle* p){
    return p->_num_active_particles;
}
static inline uint64_t LocalParticle_check_track_flag(LocalParticle* p, uint8_t index){
    return (p->track_flags >> index) & 1;
}
/* Loss reorganisation is a no-op for one particle (check_is_active never swaps). */
static inline void LocalParticle_exchange(LocalParticle*, int64_t, int64_t){}
/* at_turn is out of scope for now. increment_at_turn is compiled but never called by the bridge. */
static inline void LocalParticle_add_to_at_turn(LocalParticle*, int64_t){}
/* A map is a single particle at turn 0. The ParticlesMonitor kernel reads these. For
 * placed monitors the store slot is (at_turn - start_at_turn) with at_turn == 0. */
static inline int64_t LocalParticle_get_at_turn(LocalParticle*){ return 0; }
static inline int64_t LocalParticle_get_particle_id(LocalParticle*){ return 0; }

/* ax/ay (solenoid vector potential, set/read by the edge cancellation) are C-owned
 * variables and therefore not part of the ABI struct because they don't cross the bridge.
 * A fresh particle has ax=ay=0. Their accessors are emitted with the coord accessors
 * (LOCAL_VARS in gen_bridge.py). */

#endif /* XT_LOCAL_PARTICLE_HPP */
