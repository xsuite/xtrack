import numpy as np
import xobjects as xo


def gen_local_particle_common_src():
    src = """
    /*gpufun*/
    double LocalParticle_get_energy0( const LocalParticle *const p )
    {
        double const p0c = LocalParticle_get_p0c( p );
        double const mass0 = LocalParticle_get_mass0( p );
        return sqrt( ( p0c * p0c ) + ( mass0 * mass0 ) );
    }

    /*gpufun*/
    void LocalParticle_add_to_energy( LocalParticle* p, double const d_energy )
    {
        double delta, delta_plus_one, psigma, rvv, ptau, ptau_beta0;
        double const beta0 = LocalParticle_get_beta0( p );
        double const delta_beta0 = LocalParticle_get_delta( p ) * beta0;
        double ptau_beta0_plus_one = d_energy / LocalParticle_get_energy0( p );
        double temp = delta_beta0 * delta_beta0 +
            ( double )2.0 * delta_beta0 * beta0 + ( double )1.0;

        ptau_beta0_plus_one += sqrt( temp );
        ptau_beta0 = ptau_beta0_plus_one - ( double )1.0;
        temp = ( double )1.0 / beta0;

        ptau = ptau_beta0 * temp;
        psigma = ptau * temp;

        temp = ptau * ptau + ( double )2.0 * psigma + ( double )1.0;
        delta_plus_one = sqrt( temp );
        delta = delta_plus_one - ( double )1.0;
        rvv = delta_plus_one / ptau_beta0_plus_one;

        LocalParticle_set_delta( p, delta );
        LocalParticle_set_psigma( p, psigma );
        LocalParticle_scale_zeta( p, rvv / LocalParticle_get_rvv( p ) );
        LocalParticle_set_rvv( p, rvv );
        LocalParticle_set_rpp( p, ( double )1.0 / delta_plus_one );
    }

    /*gpufun*/
    void LocalParticle_update_delta( LocalParticle* p, double const delta_val )
    {
        double const beta0          = LocalParticle_get_beta0( p );
        double const delta_beta0    = delta_val * beta0;
        double const delta_plus_one = delta_val + ( double )1.0;

        double temp = delta_beta0 * delta_beta0 +
            ( double )2.0 * delta_beta0 * beta0 + ( double )1.0;

        double const ptau_beta0_plus_one = sqrt( temp );
        double const ptau_beta0 = ptau_beta0_plus_one - ( double )1.0;
        double const rvv = delta_plus_one / ptau_beta0_plus_one;
        temp = ( double )1.0 / beta0;

        LocalParticle_set_delta( p, delta_val );
        LocalParticle_set_psigma( p, ptau_beta0 * temp * temp );
        /* LocalParticle_scale_zeta( p, rvv / LocalParticle_get_rvv( p ) ); */
        LocalParticle_set_rpp( p, ( double )1.0 / delta_plus_one );
        LocalParticle_set_rvv( p, rvv );
    }

    /*gpufun*/
    void LocalParticle_update_p0c( LocalParticle* p, double p0c_val )
    {
        double const mass0 = LocalParticle_get_mass0( p );
        double const old_p0c = LocalParticle_get_p0c( p );
        double const old_delta = LocalParticle_get_delta( p );

        double const ppc = old_p0c * old_delta + old_p0c;
        double const new_delta = ( ppc - p0c_val ) / p0c_val;

        double const new_energy0 = sqrt(
            ( p0c_val * p0c_val ) + ( mass0 * mass0 ) );

        double const new_beta0  = p0c_val / new_energy0;
        double const new_gamma0 = new_energy0 / mass0;

        LocalParticle_set_p0c(    p, p0c_val );
        LocalParticle_set_gamma0( p, new_gamma0 );
        LocalParticle_set_beta0(  p, new_beta0 );

        LocalParticle_update_delta( p, new_delta );

        LocalParticle_scale_px( p, old_p0c / p0c_val );
        LocalParticle_scale_py( p, old_p0c / p0c_val );
    }
    """
    return src


def gen_local_particle_adapter_src():
    shared_fields = {
        "q0": {
            "type": "double",
            "api": [],
        },
        "mass0": {
            "type": "double",
            "api": [],
        },
    }

    per_part_fields = {
        "p0c": {
            "type": "double",
            "api": [],
        },
        "gamma0": {
            "type": "double",
            "api": [],
        },
        "beta0": {
            "type": "double",
            "api": ["scale"],
        },
        "s": {
            "type": "double",
            "api": ["add"],
        },
        "x": {
            "type": "double",
            "api": ["add"],
        },
        "y": {
            "type": "double",
            "api": ["add"],
        },
        "px": {
            "type": "double",
            "api": ["add", "scale"],
        },
        "py": {
            "type": "double",
            "api": ["add", "scale"],
        },
        "zeta": {
            "type": "double",
            "api": ["add", "scale"],
        },
        "delta": {
            "type": "double",
            "api": ["add", "scale"],
        },
        "psigma": {
            "type": "double",
            "api": ["add", "scale"],
        },
        "rpp": {
            "type": "double",
            "api": ["scale"],
        },
        "rvv": {
            "type": "double",
            "api": ["scale"],
        },
        "chi": {
            "type": "double",
            "api": ["scale"],
        },
        "charge_ratio": {
            "type": "double",
            "api": ["scale"],
        },
        "weight": {
            "type": "double",
            "api": ["scale"],
        },
        "particle_id": {
            "type": "int64_t",
            "api": [],
        },
        "at_element": {
            "type": "int64_t",
            "api": ["add"],
        },
        "at_turn": {
            "type": "int64_t",
            "api": ["add"],
        },
        "state": {
            "type": "int64_t",
            "api": [],
        },
        "parent_particle_id": {
            "type": "int64_t",
            "api": [],
        },
        "__rng_s1": {
            "type": "uint32_t",
            "api": [],
        },
        "__rng_s2": {
            "type": "uint32_t",
            "api": [],
        },
        "__rng_s3": {
            "type": "uint32_t",
            "api": [],
        },
        "__rng_s4": {
            "type": "uint32_t",
            "api": [],
        },
    }

    src = """
    #include <stdbool.h> //only_for_context cpu_serial cpu_openmp

    typedef struct {
        int64_t _capacity;
        int64_t _num_active_particles;
        int64_t _num_lost_particles;
        int64_t ipart;
        """
    for vv, data in shared_fields.items():
        src += (
            f"{data.get('type', 'double')} {vv};"
            + r"""
        """
        )

    for vv, data in per_part_fields.items():
        src += (
            f"/*gpuglmem*/ {data.get('type', 'double')}* {vv};"
            + r"""
        """
        )

    src += """
    }
    LocalParticle;  /* local_particle_mode == ADAPTER */

    #if !defined( XTRACK_LOCAL_PARTICLE_ADAPTER )
        #define XTRACK_LOCAL_PARTICLE_ADAPTER 0
    #endif /* !defined( XTRACK_LOCAL_PARTICLE_ADAPTER ) */

    #if !defined( XTRACK_LOCAL_PARTICLE_MODE )
        #define XTRACK_LOCAL_PARTICLE_MODE XTRACK_LOCAL_PARTICLE_ADAPTER
    #endif /* !defined( XTRACK_LOCAL_PARTICLE_MODE ) */

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*gpufun*/ int64_t LocalParticle_get__capacity(
        const LocalParticle *const p ) { return p->_capacity; }

    /*gpufun*/ int64_t LocalParticle_get__num_active_particles(
        const LocalParticle *const p ) { return p->_num_active_particles; }

    /*gpufun*/ int64_t LocalParticle_get__num_all_active_particles(
        const LocalParticle *const p ) { return p->_num_active_particles; }

    /*gpufun*/ int64_t LocalParticle_get__num_lost_particles(
        const LocalParticle *const p ) { return p->_num_lost_particles; }

    /*gpufun*/ int64_t LocalParticle_get_ipart( const LocalParticle *const p ) {
                return p->ipart; }

    """

    for vv, data in shared_fields.items():
        type_str = data.get("type_str", "double")
        api = data.get("api", [])
        src += (
            f"/*gpufun*/ {type_str} LocalParticle_get_{vv}( "
            + r"""
            const LocalParticle *const p ) { """
        )
        src += (
            f"return p->{vv}; }}"
            + r"""

    """
        )
        src += (
            f"/*gpufun*/ void LocalParticle_set_{vv}( "
            + r"""
            """
            + f"LocalParticle* p, {type_str} value ){{ "
        )
        src += (
            f"p->{vv} = value; }}"
            + r"""

    """
        )
        if "add" in api:
            src += (
                f"/*gpufun*/ void LocalParticle_add_to_{vv}( "
                + r"""
        """
                + f"LocalParticle* p, {type_str} value ){{ "
            )
            src += (
                f"p->{vv} += value; }}"
                + r"""

    """
            )
        if "scale" in api:
            src += (
                f"/*gpufun*/ void LocalParticle_scale_{vv}( "
                + r"""
        """
                + f"LocalParticle* p, {type_str} value ){{ "
            )
            src += (
                f"p->{vv} *= value; }}"
                + r"""

    """
            )

    for vv, data in per_part_fields.items():
        type_str = data.get("type_str", "double")
        api = data.get("api", [])
        src += (
            f"/*gpufun*/ {type_str} LocalParticle_get_{vv}( "
            + r"""
            const LocalParticle *const p ) { """
        )
        src += (
            f"return p->{vv}[ p->ipart ]; }}"
            + r"""

    """
        )
        src += (
            f"/*gpufun*/ void LocalParticle_set_{vv}( "
            + r"""
            """
            + f"LocalParticle* p, {type_str} value ){{ "
        )
        src += (
            f"p->{vv}[ p->ipart ] = value; }}"
            + r"""

    """
        )
        if "add" in api:
            src += (
                f"/*gpufun*/ void LocalParticle_add_to_{vv}( "
                + r"""
        """
                + f"LocalParticle* p, {type_str} value ){{ "
            )
            src += (
                f"p->{vv}[ p->ipart ] += value; }}"
                + r"""

    """
            )
        if "scale" in api:
            src += (
                f"/*gpufun*/ void LocalParticle_scale_{vv}( "
                + r"""
        """
                + f"LocalParticle* p, {type_str} value ){{ "
            )
            src += (
                f"p->{vv}[ p->ipart ] *= value; }}"
                + r"""

    """
            )

    src += r"""/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*gpufun*/ void LocalParticle_set__capacity(
        LocalParticle* p, int64_t const value ) { p->_capacity = value; }

    /*gpufun*/ void LocalParticle_set__num_active_particles(
        LocalParticle* p, int64_t const value ) { p->_num_active_particles = value; }

    /*gpufun*/ void LocalParticle_set__num_lost_particles(
        LocalParticle* p, int64_t const value ) { p->_num_lost_particles = value; }

    /*gpufun*/ void LocalParticle_set_ipart(
        LocalParticle* p, int64_t const value ) { p->ipart = value; }

    """

    src += r"""/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*gpufun*/ void LocalParticle_sync_from_particles_data(
         ParticlesData src, LocalParticle* dest )
    {
        LocalParticle_set__capacity( dest,
            ParticlesData_get__capacity( src ) );

        LocalParticle_set__num_active_particles( dest,
            ParticlesData_get__num_active_particles( src ) );

        LocalParticle_set__num_lost_particles( dest,
            ParticlesData_get__num_lost_particles( src ) );

        """
    for vv, data in shared_fields.items():
        type_str = data.get("type_str", "double")
        api = data.get("api", [])
        src += (
            f"LocalParticle_set_{vv}( dest, "
            + r"""
            """
        )
        src += (
            f"ParticlesData_get_{vv}( src ) );"
            + r"""

        """
        )

    src += r"""
    }

    /*gpufun*/ void LocalParticle_sync_to_particles_data(
        const LocalParticle *const src, ParticlesData dest,
        bool const sync_common_fields )
    {
        if( sync_common_fields ) {
            ParticlesData_set__capacity( dest,
                LocalParticle_get__capacity( src ) );

            ParticlesData_set__num_active_particles( dest,
                LocalParticle_get__num_active_particles( src ) );

            ParticlesData_set__num_lost_particles( dest,
                LocalParticle_get__num_lost_particles( src ) );

            """
    for vv, data in shared_fields.items():
        type_str = data.get("type_str", "double")
        api = data.get("api", [])
        src += (
            f"ParticlesData_set_{vv}( dest, "
            + r"""
                """
        )
        src += (
            f"LocalParticle_get_{vv}( src ) );"
            + r"""

            """
        )

    src += r"""
        }
    }

    /*gpufun*/ void LocalParticle_init_from_particles_data( ParticlesData src,
        LocalParticle* dest, int64_t const particle_index ) {

        int64_t const capacity = ParticlesData_get__capacity( src );

        if( ( particle_index >= ( int64_t )0 ) && ( capacity > particle_index ) )
        {
            dest->ipart = particle_index;
            LocalParticle_sync_from_particles_data( src, dest );

            """
    for vv, data in per_part_fields.items():
        src += (
            f"dest->{vv} = ParticlesData_getp1_{vv}( src, 0 );"
            + r"""
            """
        )
    src += r"""
        }
        else
        {
            dest->_capacity             = ( int64_t )0u;
            dest->_num_active_particles = ( int64_t )0u;
            dest->_num_lost_particles   = ( int64_t )0u;
            dest->ipart                 = ( int64_t )0u;

            dest->p0c                   = NULL;
            dest->gamma0                = NULL;
            dest->beta0                 = NULL;
            dest->s                     = NULL;
            dest->x                     = NULL;
            dest->y                     = NULL;
            dest->px                    = NULL;
            dest->py                    = NULL;
            dest->zeta                  = NULL;
            dest->psigma                = NULL;
            dest->delta                 = NULL;
            dest->rpp                   = NULL;
            dest->rvv                   = NULL;
            dest->chi                   = NULL;
            dest->charge_ratio          = NULL;
            dest->weight                = NULL;

            dest->particle_id           = NULL;
            dest->at_element            = NULL;
            dest->at_turn               = NULL;
            dest->state                 = NULL;
            dest->parent_particle_id    = NULL;

            dest->__rng_s1              = NULL;
            dest->__rng_s2              = NULL;
            dest->__rng_s3              = NULL;
            dest->__rng_s4              = NULL;
        }
    }

    /*gpufun*/ void LocalParticle_to_particles_data(
        const LocalParticle *const src, ParticlesData dest,
        int64_t const dest_particle_index,
        bool const copy_common_fields ) {

        LocalParticle_sync_to_particles_data( src, dest, copy_common_fields );

        """
    for vv, data in per_part_fields.items():
        src += (
            f"ParticlesData_set_{vv}( dest, dest_particle_index,"
            + r"""
            """
        )
        src += (
            f"LocalParticle_get_{vv}( src ) );"
            + r"""

        """
        )

    src += r"""
    }

    /*gpufun*/ void LocalParticle_exchange( LocalParticle* p,
        int64_t const src_idx, int64_t const dest_idx ) {

        int64_t  temp_int64_t;
        uint32_t temp_uint32_t;
        double   temp_double;

        ( void )temp_int64_t;
        ( void )temp_uint32_t;
        ( void )temp_double;

        if( src_idx == dest_idx ) return;

        """
    for vv, data in per_part_fields.items():
        type_str = data.get("type", "double")
        src += (
            f"temp_{type_str} = p->{vv}[ dest_idx ];"
            + r"""
        """
        )
        src += (
            f"p->{vv}[ dest_idx ] = p->{vv}[ src_idx ];"
            + r"""
        """
        )
        src += (
            f"p->{vv}[ src_idx  ] = p->{vv}[ dest_idx ];"
            + r"""

        """
        )
    src += r"""
    }

    /*gpufun*/ bool LocalParticle_is_active( const LocalParticle *const p ) {
        return ( p->state[ p->ipart ] == 1 ); }

    /*gpufun*/ bool LocalParticle_are_any_active( LocalParticle* p ) {
        #if defined( CPUIMPL )
        int64_t n_active = p->_num_active_particles;
        int64_t n_lost = p->_num_lost_particles;
        int64_t ipart = 0;

        while( ipart < n_active ) {
            if( part->state[ ipart ] != 1 ) {
                LocalParticle_exchange( p, ipart, n_active - 1 );
                --n_active;
                ++n_lost; }
            else { ++ipart; } }

        p->_num_active_particles = n_active;
        p->_num_lost_particles = n_lost;
        return ( p->_num_active_particles > 0 );

        #else /* !defined( CPUIMPL ) */
        bool has_any_active = false;
        int64_t const n_part = p->_capacity;
        int64_t ipart = 0;

        while( ( !has_any_active ) && ( ipart < n_part ) ) {
            has_any_active = ( p->state[ ipart++ ] == 1 ); }

        return has_any_active;

        #endif /* defined( CPUIMPL ) */
    }

    /*gpufun*/ bool LocalParticle_is_lost( const LocalParticle *const p ) {
        return ( p->state[ p->ipart ] != 1 ); }

    /*gpufun*/ void LocalParticle_mark_as_lost( LocalParticle* p ) {
        p->state[ p->ipart ] = 0; }
    """
    return src


def gen_local_particle_local_copy_src():
    shared_fields = {
        "q0": {
            "type": "double",
            "api": [],
        },
        "mass0": {
            "type": "double",
            "api": [],
        },
    }

    per_part_fields = {
        "p0c": {
            "type": "double",
            "api": [],
        },
        "gamma0": {
            "type": "double",
            "api": [],
        },
        "beta0": {
            "type": "double",
            "api": ["scale"],
        },
        "s": {
            "type": "double",
            "api": ["add"],
        },
        "x": {
            "type": "double",
            "api": ["add"],
        },
        "y": {
            "type": "double",
            "api": ["add"],
        },
        "px": {
            "type": "double",
            "api": ["add", "scale"],
        },
        "py": {
            "type": "double",
            "api": ["add", "scale"],
        },
        "zeta": {
            "type": "double",
            "api": ["add", "scale"],
        },
        "delta": {
            "type": "double",
            "api": ["add", "scale"],
        },
        "psigma": {
            "type": "double",
            "api": ["add", "scale"],
        },
        "rpp": {
            "type": "double",
            "api": ["scale"],
        },
        "rvv": {
            "type": "double",
            "api": ["scale"],
        },
        "chi": {
            "type": "double",
            "api": ["scale"],
        },
        "charge_ratio": {
            "type": "double",
            "api": ["scale"],
        },
        "weight": {
            "type": "double",
            "api": ["scale"],
        },
        "state_particle_id": {
            "type": "int64_t",
            "api": [],
        },
        "at_element": {
            "type": "int64_t",
            "api": ["add"],
        },
        "at_turn": {
            "type": "int64_t",
            "api": ["add"],
        },
        "parent_particle_id": {
            "type": "int64_t",
            "api": [],
        },
        "__rng_s1": {
            "type": "uint32_t",
            "api": [],
        },
        "__rng_s2": {
            "type": "uint32_t",
            "api": [],
        },
        "__rng_s3": {
            "type": "uint32_t",
            "api": [],
        },
        "__rng_s4": {
            "type": "uint32_t",
            "api": [],
        },
    }

    src = """
    #include <stdbool.h> //only_for_context cpu_serial cpu_openmp

    typedef struct {
        int64_t  _capacity;
        int64_t  ipart;
        """
    for vv, data in shared_fields.items():
        src += (
            f"{data.get( 'type', 'double' )} {vv};"
            + r"""
        """
        )

    for vv, data in per_part_fields.items():
        src += (
            f"{data.get( 'type', 'double' )} {vv};"
            + r"""
        """
        )
    src += r"""
    }
    LocalParticle; /* local_particle_mode == THREAD_LOCAL_COPY */

    #if !defined( XTRACK_LOCAL_PARTICLE_THREAD_LOCAL_COPY )
        #define XTRACK_LOCAL_PARTICLE_THREAD_LOCAL_COPY 1
    #endif /* !defined( XTRACK_LOCAL_PARTICLE_THREAD_LOCAL_COPY ) */

    #if !defined( XTRACK_LOCAL_PARTICLE_MODE )
        #define XTRACK_LOCAL_PARTICLE_MODE XTRACK_LOCAL_PARTICLE_THREAD_LOCAL_COPY
    #endif /* !defined( XTRACK_LOCAL_PARTICLE_MODE ) */

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*gpufun*/ int64_t LocalParticle_get__capacity(
        const LocalParticle *const p ) { return p->_capacity; }

    /*gpufun*/ int64_t LocalParticle_get__num_all_active_particles(
        const LocalParticle *const p ) { return p->_capacity; }

    """

    for vv, data in shared_fields.items():
        type_str = data.get("type", "double")
        api = data.get("api", [])
        src += (
            f"/*gpufun*/ {type_str} LocalParticle_get_{vv}( "
            + r"""
        """
            + f" const LocalParticle *const p ) {{ return p->{vv}; }}"
            + r"""

    """
        )
        src += (
            f"/*gpufun*/ void LocalParticle_set_{vv}( "
            + r"""
        """
            + f" LocalParticle* p, {type_str} value ) {{ p->{vv} = value; }}"
        )
        src += r"""

    """
        if "add" in api:
            src += (
                f"/*gpufun*/ void LocalParticle_add_to_{vv}( "
                + r"""
        """
                + f" LocalParticle* p, {type_str} value ) {{p->{vv} += value; }}"
            )
            src += r"""

    """
        if "scale" in api:
            src += (
                f"/*gpufun*/ void LocalParticle_scale_{vv}( "
                + r"""
        """
                + f" LocalParticle* p, {type_str} value ) {{p->{vv} *= value; }}"
            )
            src += r"""

    """

    for vv, data in per_part_fields.items():
        type_str = data.get("type", "double")
        api = data.get("api", [])
        src += (
            f"/*gpufun*/ {type_str} LocalParticle_get_{vv}( "
            + r"""
        """
            + f" const LocalParticle *const p ) {{ return p->{vv}; }}"
            + r"""

    """
        )
        src += (
            f"/*gpufun*/ void LocalParticle_set_{vv}( "
            + r"""
        """
            + f" LocalParticle* p, {type_str} value ) {{ p->{vv} = value; }}"
        )
        src += r"""

    """
        if "add" in api:
            src += (
                f"/*gpufun*/ void LocalParticle_add_to_{vv}( "
                + r"""
        """
                + f" LocalParticle* p, {type_str} value ) {{ p->{vv} += value; }}"
            )
            src += r"""

    """
        if "scale" in api:
            src += (
                f"/*gpufun*/ void LocalParticle_scale_{vv}( "
                + r"""
            """
                + f" LocalParticle* p, {type_str} value ) {{ p->{vv} *= value; }}"
            )
            src += r"""

    """

    src += r"""
    /*gpufun*/ int64_t LocalParticle_get_state( const LocalParticle *const p ) {
        int64_t const temp = LocalParticle_get_state_particle_id( p );
        return ( temp >= 0 ) ? ( int64_t )1 : ( int64_t )0; }

    /*gpufun*/ void LocalParticle_set_state( LocalParticle* p, int64_t value ) {
        int64_t state_particle_id = LocalParticle_get_state_particle_id( p );
        if( state_particle_id < 0 )
            state_particle_id = ( -state_particle_id ) - ( int64_t )1;
        if( value != ( int64_t )1 )
            state_particle_id = -( state_particle_id + ( int64_t )1 );
        LocalParticle_set_state_particle_id( p, state_particle_id ); }

    /*gpufun*/ int64_t LocalParticle_get_particle_id( const LocalParticle *const p ) {
            int64_t state_particle_id = LocalParticle_get_state_particle_id( p );
            return ( state_particle_id >= 0 )
                ?  state_particle_id : ( -state_particle_id ) - ( int64_t )1; }

    /*gpufun*/ void LocalParticle_set_particle_id(
        LocalParticle* p, int64_t const particle_id ) {
        int64_t const state_particle_id = ( LocalParticle_get_state( p ) == 1 )
            ? particle_id : -( particle_id + ( int64_t )1 );
        LocalParticle_set_state_particle_id( p, state_particle_id ); }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*gpufun*/ void LocalParticle_sync_from_particles_data(
         ParticlesData src, LocalParticle* dest )
    {
        int64_t state_particle_id =
            ParticlesData_get_particle_id( src, dest->ipart );

        if( ParticlesData_get_state( src, dest->ipart ) != 1 )
            state_particle_id = -( state_particle_id + ( int64_t )1 );

        dest->state_particle_id = state_particle_id;
        dest->_capacity = ParticlesData_get__capacity( src );
        """

    for vv, data in shared_fields.items():
        src += (
            f"dest->{vv} = ParticlesData_get_{vv}( src );"
            + r"""
        """
        )

    for vv, data in per_part_fields.items():
        if vv == "state_particle_id":
            continue
        src += (
            f"dest->{vv} = ParticlesData_get_{vv}( src, dest->ipart );"
            + r"""
        """
        )

    src += r"""
    }

    /*gpufun*/ void LocalParticle_sync_to_particles_data(
        const LocalParticle *const src, ParticlesData dest,
        bool const sync_common_fields )
    {
        if( sync_common_fields ) {
            ParticlesData_set__capacity( dest,
                LocalParticle_get__capacity( src ) );

            """

    for vv, data in shared_fields.items():
        src += (
            f"ParticlesData_set_{vv}( "
            + r"""
                """
            + f"dest, LocalParticle_get_{vv}( src ) );"
            + r"""

            """
        )

    src += r"""
        }

        ParticlesData_set_state( dest, src->ipart,
            LocalParticle_get_state( src ) );

        ParticlesData_set_particle_id( dest, src->ipart,
            LocalParticle_get_particle_id( src ) );

        """

    for vv, data in per_part_fields.items():
        if vv == "state_particle_id":
            continue
        src += (
            f"ParticlesData_set_{vv}( dest, src->ipart, src->{vv} );"
            + r"""

        """
        )
    src += r"""

    }

    /*gpufun*/ void LocalParticle_init_from_particles_data( ParticlesData src,
        LocalParticle* dest, int64_t const particle_index )
    {
        int64_t const capacity = ParticlesData_get__capacity( src );

        if( ( particle_index >= ( int64_t )0 ) && ( capacity > particle_index ) )
        {
            dest->ipart = particle_index;
            LocalParticle_sync_from_particles_data( src, dest );
        }
        else
        {
            dest->_capacity = ( int64_t )0;
            dest->ipart = ( int64_t )0;

            """
    for vv, data in shared_fields.items():
        type_str = data.get("type", "double")
        val = "0.0" if type_str == "double" else "0"
        src += (
            f"dest->{vv} = ( {type_str} ){val};"
            + r"""
            """
        )

    src += r"""
            """
    for vv, data in per_part_fields.items():
        type_str = data.get("type", "double")
        val = "0.0" if type_str == "double" else "0"
        src += (
            f"dest->{vv} = ( {type_str} ){val};"
            + r"""
            """
        )
    src += r"""

        }
    }

    /*gpufun*/ void LocalParticle_to_particles_data(
        LocalParticle* src, ParticlesData dest,
        int64_t const dest_particle_index,
        bool const copy_common_fields ) {

        int64_t const saved_ipart = src->ipart;
        src->ipart = dest_particle_index;
        LocalParticle_sync_to_particles_data( src, dest, copy_common_fields );
        src->ipart = saved_ipart;
    }

    /*gpufun*/ void LocalParticle_exchange( LocalParticle* p,
        int64_t const src_idx, int64_t const dest_idx )
    {
        ( void )p;
        ( void )src_idx;
        ( void )dest_idx;
    }

    /*gpufun*/ bool LocalParticle_is_active( const LocalParticle *const p ) {
        return ( p->state_particle_id >= 0 ); }

    /*gpufun*/ bool LocalParticle_are_any_active( LocalParticle* p ) {
        return ( p->state_particle_id >= 0 ); }

    /*gpufun*/ bool LocalParticle_is_lost( const LocalParticle *const p ) {
        return ( p->state_particle_id < 0 ); }

    /*gpufun*/ void LocalParticle_mark_as_lost( LocalParticle* p ) {
        if( p->state_particle_id >= 0 )
            p->state_particle_id = -( p->state_particle_id +  ( int64_t )1 ); }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*gpufun*/ int64_t LocalParticle_get_ipart(
        const LocalParticle *const p ) { return p->ipart; }

    /*gpufun*/ int64_t LocalParticle_get__num_active_particles(
        const LocalParticle *const p ) { return ( LocalParticle_is_active( p ) )
            ? ( int64_t )1 : ( int64_t )0; }

    /*gpufun*/ int64_t LocalParticle_get__num_lost_particles(
        const LocalParticle *const p ) { return ( LocalParticle_is_lost( p ) )
            ? ( int64_t )1 : ( int64_t )0; }

    """
    return src


def gen_local_particle_shared_copy_src():
    shared_fields = {
        "q0": {
            "offset": 0,
            "type": "double",
            "api": [],
        },
        "mass0": {
            "offset": 8,
            "type": "double",
            "api": [],
        },
    }

    per_part_fields = {
        "p0c": {
            "offset": 0,
            "type": "double",
            "api": [],
        },
        "gamma0": {
            "offset": 8,
            "type": "double",
            "api": [],
        },
        "beta0": {
            "offset": 16,
            "type": "double",
            "api": ["scale"],
        },
        "s": {
            "offset": 24,
            "type": "double",
            "api": ["add"],
        },
        "x": {
            "offset": 32,
            "type": "double",
            "api": ["add"],
        },
        "y": {
            "offset": 40,
            "type": "double",
            "api": ["add"],
        },
        "px": {
            "offset": 48,
            "type": "double",
            "api": ["add", "scale"],
        },
        "py": {
            "offset": 56,
            "type": "double",
            "api": ["add", "scale"],
        },
        "zeta": {
            "offset": 64,
            "type": "double",
            "api": ["add", "scale"],
        },
        "delta": {
            "offset": 72,
            "type": "double",
            "api": ["add", "scale"],
        },
        "psigma": {
            "offset": 80,
            "type": "double",
            "api": ["add", "scale"],
        },
        "rpp": {
            "offset": 88,
            "type": "double",
            "api": ["scale"],
        },
        "rvv": {
            "offset": 96,
            "type": "double",
            "api": ["scale"],
        },
        "chi": {
            "offset": 104,
            "type": "double",
            "api": ["scale"],
        },
        "charge_ratio": {
            "offset": 112,
            "type": "double",
            "api": ["scale"],
        },
        "weight": {
            "offset": 120,
            "type": "double",
            "api": ["scale"],
        },
        "state_particle_id": {
            "offset": 128,
            "type": "int64_t",
        },
    }

    priv_part_fields = {
        "at_element": {
            "type": "int64_t",
            "api": [
                "add",
            ],
        },
        "at_turn": {
            "type": "int64_t",
            "api": [
                "add",
            ],
        },
        "parent_particle_id": {
            "type": "int64_t",
            "api": [],
        },
        "__rng_s1": {
            "type": "uint32_t",
            "api": [],
        },
        "__rng_s2": {
            "type": "uint32_t",
            "api": [],
        },
        "__rng_s3": {
            "type": "uint32_t",
            "api": [],
        },
        "__rng_s4": {
            "type": "uint32_t",
            "api": [],
        },
    }

    src = """
    #include <stdbool.h> //only_for_context cpu_serial cpu_openmp

    typedef struct {
        /*gpusharedmem*/ char*  special_fields;
        uint32_t local_per_particle_offset;
        uint32_t local_common_offset;

        int64_t  _capacity;
        int64_t  global_ipart;
        """
    for vv, data in priv_part_fields.items():
        src += (
            f"{data.get( 'type', 'int64_t' )}  {vv};"
            + r"""
        """
        )

    src += r"""
    }
    LocalParticle;  /* local_particle_mode == SHARED_COPY */

    #if !defined( XTRACK_LOCAL_PARTICLE_SHARED_COPY )
        #define XTRACK_LOCAL_PARTICLE_SHARED_COPY 2
    #endif /* !defined( XTRACK_LOCAL_PARTICLE_SHARED_COPY ) */

    #if !defined( XTRACK_LOCAL_PARTICLE_MODE )
        #define XTRACK_LOCAL_PARTICLE_MODE XTRACK_LOCAL_PARTICLE_SHARED_COPY
    #endif /* !defined( XTRACK_LOCAL_PARTICLE_MODE ) */

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*gpufun*/ uint32_t LocalParticle_get_local_per_particle_offset(
        const LocalParticle *const p ) { return p->local_per_particle_offset; }

    /*gpufun*/ uint32_t LocalParticle_get_local_common_offset(
        const LocalParticle *const p ) { return p->local_common_offset; }

    /*gpufun*/ uint32_t LocalParticle_get_local_pitch( const LocalParticle *const p ) {
        ( void )p;
        return sizeof( double ) * sizeof( char ) * 17u; }

    """

    for vv, data in shared_fields.items():
        type_str = data["type"]
        offset = data["offset"]
        api = data.get("api", [])

        src += f"/*gpufun*/ {type_str} LocalParticle_get_{vv}( "
        src += f"const LocalParticle *const p ){{"
        src += r"""
            """
        src += f"return *( ( /*gpusharedmem*/ const double *const )( "
        src += r"""
                """
        src += f"p->special_fields + p->local_common_offset + {offset} ) ); }}"
        src += r"""

    """
        src += f"/*gpufun*/ void LocalParticle_set_{vv}( LocalParticle* p, {type_str} value ){{"
        src += r"""
            """
        src += (
            f"*( ( /*gpusharedmem*/ {type_str}* )( "
            + r"""
                """
            + f"p->special_fields + p->local_common_offset + {offset} )"
        )
        src += r"""
                ) = value; }

    """

        if "add" in api:
            src += r"""
    /*gpufun*/ """
            src += f"void LocalParticle_add_to_{vv}( LocalParticle* p, {type_str} value ){{"
            src += r"""
                """
            src += (
                f"*( ( /*gpusharedmem*/ {type_str}* )( "
                + r"""
                    """
                + f"p->special_fields + p->local_common_offset + {offset} )"
            )
            src += r"""
                    ) += value; }

    """

        if "scale" in api:
            src += r"""
    /*gpufun*/ """
            src += (
                f"void LocalParticle_scale_{vv}( LocalParticle* p, {type_str} value ){{"
            )
            src += r"""
                """
            src += (
                f"*( ( /*gpusharedmem*/ {type_str}* )( "
                + r"""
                    """
                + f"p->special_fields + p->local_common_offset + {offset} )"
            )
            src += r"""
                    ) *= value; }

    """

    for vv, data in per_part_fields.items():
        type_str = data.get("type", "double")
        offset = data.get("offset", 0)
        api = data.get("api", [])
        src += r"""
    """
        src += f"/*gpufun*/ {type_str} LocalParticle_get_{vv}( "
        src += f"const LocalParticle *const p ){{"
        src += r"""
            """
        src += f"return *( ( /*gpusharedmem*/ const double *const )( "
        src += r"""
                """
        src += f"p->special_fields + p->local_per_particle_offset + {offset} ) ); }}"
        src += r"""

    """
        src += f"/*gpufun*/ void LocalParticle_set_{vv}( LocalParticle* p, {type_str} value ){{"
        src += r"""
            """
        src += (
            f"*( ( /*gpusharedmem*/ {type_str}* )( "
            + r"""
                """
            + f"p->special_fields + p->local_per_particle_offset + {offset} )"
        )
        src += r"""
                ) = value; }

    """

        if "add" in api:
            src += f"/*gpufun*/ void LocalParticle_add_to_{vv}( LocalParticle* p, {type_str} value ){{"
            src += r"""
            """
            src += (
                f"*( ( /*gpusharedmem*/ {type_str}* )( "
                + r"""
                """
                + f"p->special_fields + p->local_per_particle_offset + {offset} )"
            )
            src += r"""
                    ) += value; }

    """

        if "scale" in api:
            src += f"/*gpufun*/ void LocalParticle_scale_{vv}( LocalParticle* p, {type_str} value ){{"
            src += r"""
                """
            src += (
                f"*( ( /*gpusharedmem*/ {type_str}* )( "
                + r"""
                """
                + f"p->special_fields + p->local_per_particle_offset + {offset} )"
            )
            src += r"""
                    ) *= value; }

    """

    src += r"""
    /*gpufun*/ int64_t LocalParticle_get_state( const LocalParticle *const p ) {
        int64_t const temp = LocalParticle_get_state_particle_id( p );
        return ( temp >= 0 ) ? ( int64_t )1 : ( int64_t )0; }

    /*gpufun*/ void LocalParticle_set_state( LocalParticle* p, int64_t value ) {
        int64_t state_particle_id = LocalParticle_get_state_particle_id( p );
        if( state_particle_id < 0 )
            state_particle_id = -state_particle_id - ( int64_t )1;
        if( value != ( int64_t )1 )
            state_particle_id = -( state_particle_id + ( int64_t )1 );
        LocalParticle_set_state_particle_id( p, state_particle_id ); }

    /*gpufun*/ int64_t LocalParticle_get_particle_id( const LocalParticle *const p ) {
            int64_t state_particle_id = LocalParticle_get_state_particle_id( p );
            return ( state_particle_id >= 0 )
                ?  state_particle_id : -state_particle_id - ( int64_t )1; }

    /*gpufun*/ void LocalParticle_set_particle_id(
        LocalParticle* p, int64_t const particle_id ) {
        int64_t const state_particle_id = ( LocalParticle_get_state( p ) == 1 )
            ? particle_id : -( particle_id + ( int64_t )1 );
        LocalParticle_set_state_particle_id( p, state_particle_id ); }

    """

    for vv, data in priv_part_fields.items():
        type_str = data.get("type", "double")
        api = data.get("api", [])
        src += (
            f"/*gpufun*/ {type_str} LocalParticle_get_{vv}( "
            + r"""
        """
            + f"const LocalParticle *const p ) {{ return p->{vv}; }}"
            + r"""

    """
        )
        src += (
            f"/*gpufun*/ void LocalParticle_set_{vv}( "
            + r"""
        """
            + f"LocalParticle* p, {type_str} value ) {{ p->{vv} = value; }}"
            + r"""

    """
        )
        if "add" in api:
            src += (
                f"/*gpufun*/ void LocalParticle_add_to_{vv}( "
                + r"""
        """
                + f"LocalParticle* p, {type_str} value ) {{ p->{vv} += value; }}"
                + r"""

    """
            )
        if "scale" in api:
            src += (
                f"/*gpufun*/ void LocalParticle_scale_{vv}( "
                + r"""
        """
                + f"LocalParticle* p, {type_str} value ) {{ p->{vv} *= value; }}"
                + r"""

    """
            )

    src += r"""
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*gpufun*/ void LocalParticle_sync_from_particles_data(
         ParticlesData src, LocalParticle* dest )
    {
        int64_t state_particle_id = ParticlesData_get_particle_id(
            src, dest->global_ipart );

        if( ParticlesData_get_state( src, dest->global_ipart ) != 1 )
            state_particle_id = -state_particle_id;
        LocalParticle_set_state_particle_id( dest, state_particle_id );
        """

    for vv, data in shared_fields.items():
        src += f"LocalParticle_set_{vv}( dest, ParticlesData_get_{vv}( src ) );"
        src += r"""
        """

    for vv, data in per_part_fields.items():
        if vv == "state_particle_id":
            continue
        src += (
            f"LocalParticle_set_{vv}( dest, ParticlesData_get_{vv}( "
            + r"""
                src, dest->global_ipart ) );
        """
        )

    for vv, data in priv_part_fields.items():
        src += (
            f"LocalParticle_set_{vv}( dest, ParticlesData_get_{vv}( "
            + r"""
                src, dest->global_ipart ) );
        """
        )

    src += r"""
    }

    /*gpufun*/ void LocalParticle_sync_to_particles_data(
        const LocalParticle *const src, ParticlesData dest,
        bool const sync_common_fields )
    {
        """

    for vv, data in per_part_fields.items():
        if vv == "state_particle_id":
            continue
        src += (
            f"ParticlesData_set_{vv}( dest, src->global_ipart, "
            + r"""
                """
            + f"LocalParticle_get_{vv}( src ) );"
            + r"""
        """
        )

    src += r"""
        """

    for vv, data in priv_part_fields.items():
        src += (
            f"ParticlesData_set_{vv}( dest, src->global_ipart, "
            + r"""
                """
            + f"LocalParticle_get_{vv}( src ) );"
            + r"""
        """
        )

    src += r"""
        ParticlesData_set_particle_id( dest, src->global_ipart,
            LocalParticle_get_particle_id( src ) );

        ParticlesData_set_state( dest, src->global_ipart,
            LocalParticle_get_state( src ) );

        if( sync_common_fields )
        {
            """

    for vv, data in shared_fields.items():
        src += f"ParticlesData_set_{vv}( dest, LocalParticle_get_{vv}( src ) );"
        src += r"""
            """

    src += r"""
        }
    }

    /*gpufun*/ void LocalParticle_init_from_particles_data( ParticlesData src,
        LocalParticle* dest, int64_t const particle_index,
        /*gpusharedmem*/ char* special_fields )
    {
        if( ( particle_index >= 0 ) && ( special_fields != NULL ) )
        {
            uint32_t const local_ipart = 0; //only_for_context cpu_serial
            uint32_t const local_ipart = 0; //only_for_context cpu_openmp
            uint32_t const local_ipart = get_local_id( 0 ); //only_for_context opencl
            uint32_t const local_ipart = threadIdx.x; //only_for_context cuda

            uint32_t const local_n_part = ParticlesData_get__capacity( src ); //only_for_context cpu_serial cpu_openmp
            uint32_t const local_n_part = get_local_size( 0 ); //only_for_context opencl
            uint32_t const local_n_part = blockDim.x; //only_for_context cuda
            uint32_t const local_pitch  = LocalParticle_get_local_pitch( dest );

            dest->local_per_particle_offset = local_ipart  * local_pitch;
            dest->local_common_offset       = local_n_part * local_pitch;
            dest->special_fields            = special_fields;
            dest->global_ipart              = particle_index;
            dest->_capacity                 = ParticlesData_get__capacity( src );
            LocalParticle_sync_from_particles_data( src, dest );
        }
        else
        {
            dest->local_per_particle_offset = ( uint32_t )0u;
            dest->local_common_offset = ( uint32_t )0u;
            dest->special_fields = NULL;
            dest->global_ipart = ( int64_t )-1;
            dest->_capacity = ( int64_t )0u;
        }
    }

    /*gpufun*/ void LocalParticle_to_particles_data(
        LocalParticle* src, ParticlesData dest,
        int64_t const dest_particle_index, bool const copy_common_fields ) {
        int64_t const saved_ipart = src->global_ipart;
        src->global_ipart = dest_particle_index;
        LocalParticle_sync_to_particles_data( src, dest, copy_common_fields );
        src->global_ipart = saved_ipart; }

    /*gpufun*/ void LocalParticle_exchange( LocalParticle* p,
        int64_t const src_idx, int64_t const dest_idx ) {
        ( void )p;
        ( void )src_idx;
        ( void )dest_idx; }

    /*gpufun*/ bool LocalParticle_is_active( const LocalParticle *const p ) {
        return ( *( ( /*gpusharedmem*/ const int64_t *const )(
            p->special_fields + p->local_per_particle_offset + """

    src += f"{per_part_fields[ 'state_particle_id' ][ 'offset' ]} ) ) >= 0 ); }}"
    src += r"""

    /*gpufun*/ bool LocalParticle_are_any_active( LocalParticle* p ) {
        return LocalParticle_is_active( p ); }

    /*gpufun*/ bool LocalParticle_is_lost( const LocalParticle *const p ) {
        return !( LocalParticle_is_active( p ) ); }

    /*gpufun*/ void LocalParticle_mark_as_lost( LocalParticle* p ) {
        /*gpusharedmem*/ int64_t* ptr = ( /*gpusharedmem*/ int64_t* )(
            p->special_fields + p->local_per_particle_offset + """
    src += f" {per_part_fields[ 'state_particle_id' ][ 'offset' ]} );"
    src += r"""
        int64_t const cur_state_particle_id = *ptr;
        if( cur_state_particle_id >= 0 ) *ptr = -( cur_state_particle_id + ( int64_t )1 ); }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*gpufun*/ int64_t LocalParticle_get_global_ipart(
        const LocalParticle *const p ) { return p->global_ipart; }

    /*gpufun*/ int64_t LocalParticle_get__capacity(
        const LocalParticle *const p ) { return p->_capacity; }

    /*gpufun*/ int64_t LocalParticle_get__num_all_active_particles(
        const LocalParticle *const p ) { return p->_capacity; }
    """
    return src
