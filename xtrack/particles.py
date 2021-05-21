import pysixtrack
import numpy as np
import xobjects as xo

from .dress import dress

pmass = 938.2720813e6


scalar_vars = (
    (xo.Int64,   'num_particles'),
    (xo.Float64, 'q0'),
    (xo.Float64, 'mass0'),
    (xo.Float64, 'beta0'),
    (xo.Float64, 'gamma0'),
    (xo.Float64, 'p0c',)
    )

per_particle_vars = [
    (xo.Float64, 's'),
    (xo.Float64, 'x'),
    (xo.Float64, 'y'),
    (xo.Float64, 'px'),
    (xo.Float64, 'py'),
    (xo.Float64, 'zeta'),
    (xo.Float64, 'psigma'),
    (xo.Float64, 'delta'),
    (xo.Float64, 'rpp'),
    (xo.Float64, 'rvv'),
    (xo.Float64, 'chi'),
    (xo.Float64, 'charge_ratio'),
    (xo.Float64, 'weight'),
    (xo.Int64, 'particle_id'),
    (xo.Int64, 'at_element'),
    (xo.Int64, 'at_turn'),
    (xo.Int64, 'state'),
    ]

fields = {}
for tt, nn in scalar_vars:
    fields[nn] = tt

for tt, nn in per_particle_vars:
    fields[nn] = tt[:]

ParticlesData = type(
        'ParticlesData',
        (xo.Struct,),
        fields)

class Particles(dress(ParticlesData)):

    def __init__(self, pysixtrack_particles=None, num_particles=None, **kwargs):

        # Initalize array sizes
        if pysixtrack_particles is not None:
            if hasattr(pysixtrack_particles, '__iter__'):
                # Assuming list of pysixtrack particles
                num_particles = len(pysixtrack_particles)
            else:
                num_particles = len(pysixtrack_particles.x)
        else:
            assert num_particles is not None

        kwargs.update(
                {kk: np.arange(num_particles)+1 for tt, kk in per_particle_vars})
        kwargs['num_particles'] = num_particles

        if '_context' in kwargs.keys():
            ctx = kwargs['_context']
            for kk, vv in kwargs.items():
                if isinstance(vv, np.ndarray):
                    kwargs[kk] = ctx.nparray_to_context_array(vv)

        self.xoinitialize(**kwargs)

        # Initalize arrays
        if pysixtrack_particles is not None:
            if hasattr(pysixtrack_particles, '__iter__'):
                for ii, pyst_part in enumerate(pysixtrack_particles):
                    self.set_particles_from_pysixtrack(ii, pyst_part,
                                                   set_scalar_vars=(ii==0))
            else:
                self.set_particles_from_pysixtrack(None,
                                                   pysixtrack_particles,
                                                   set_scalar_vars=True)


    def _set_p0c(self):
        energy0 = np.sqrt(self.p0c ** 2 + self.mass0 ** 2)
        self.beta0 = self.p0c / energy0
        self.gamma0 = energy0 / self.mass0

    def _set_delta(self):
        rep = np.sqrt(self.delta ** 2 + 2 * self.delta + 1 / self.beta0 ** 2)
        irpp = 1 + self.delta
        self.rpp = 1 / irpp
        beta = irpp / rep
        self.rvv = beta / self.beta0
        self.psigma = (
            np.sqrt(self.delta ** 2 + 2 * self.delta + 1 / self.beta0 ** 2)
            / self.beta0
            - 1 / self.beta0 ** 2
        )

    @property
    def ptau(self):
        return (
            np.sqrt(self.delta ** 2 + 2 * self.delta + 1 / self.beta0 ** 2)
            - 1 / self.beta0
        )

    def set_reference(self, p0c=7e12, mass0=pmass, q0=1):
        self.q0 = q0
        self.mass0 = mass0
        self.p0c = p0c
        return self

    def set_particles_from_pysixtrack(self, index, pysixtrack_particle,
            set_scalar_vars=False, check_scalar_vars=True):

        context = self._buffer.context

        if not(set_scalar_vars) and check_scalar_vars:
            for tt, vv in scalar_vars:
                if vv == 'num_particles':
                    continue
                vv_first = getattr(self, vv)
                assert (getattr(pysixtrack_particle, vv)
                               == vv_first), f'Inconsistent "{vv}"'

        if set_scalar_vars:
            for tt, vv in scalar_vars:
                if vv == 'num_particles':
                    continue
                setattr(self, vv, getattr(pysixtrack_particle, vv))

        for tt, vv in per_particle_vars:
            if vv == 'weight':
                if not hasattr(pysixtrack_particle, 'weight'):
                    if index is None:
                        self.weight = np.zeros(int(self.num_particles)) + 1.
                    else:
                        self.weight[index] = 1.
                    continue

            if vv == 'mass_ratio':
                vv_pyst = 'mratio'
            elif vv == 'charge_ratio':
                vv_pyst = 'qratio'
            elif vv == 'particle_id':
                vv_pyst = 'partid'
            elif vv == 'at_element':
                vv_pyst = 'elemid'
            elif vv == 'at_turn':
                vv_pyst = 'turn'
            else:
                vv_pyst = vv

            if index is None:
                val_pyst = getattr(pysixtrack_particle, vv_pyst)
                if np.isscalar(val_pyst):
                    val_pyst = val_pyst + np.zeros(int(self.num_particles))
                setattr(self, vv, context.nparray_to_context_array(val_pyst))
            else:
                getattr(self, vv)[index] = getattr(
                            pysixtrack_particle, vv_pyst)

def gen_local_particle_api(mode='no_local_copy'):

    if mode != 'no_local_copy':
        raise NotImplementedError

    src_lines = []
    src_lines.append('''typedef struct{''')
    for tt, vv in scalar_vars:
        src_lines.append('                 ' + tt._c_type + '  '+vv+';')
    for tt, vv in per_particle_vars:
        src_lines.append('    /*gpuglmem*/ ' + tt._c_type + '* '+vv+';')
    src_lines.append(    '                 int64_t ipart;')
    src_lines.append('}LocalParticle;')
    src_typedef = '\n'.join(src_lines)

    src_lines = []
    src_lines.append('''
    /*gpufun*/
    void Particles_to_LocalParticle(ParticlesData source,
                                    LocalParticle* dest,
                                    int64_t id){''')
    for tt, vv in scalar_vars:
        src_lines.append(
                f'  dest->{vv} = ParticlesData_get_'+vv+'(source);')
    for tt, vv in per_particle_vars:
        src_lines.append(
                f'  dest->{vv} = ParticlesData_getp1_'+vv+'(source, 0);')
    src_lines.append('  dest->ipart = id;')
    src_lines.append('}')
    src_particles_to_local = '\n'.join(src_lines)

    src_lines=[]
    for tt, vv in per_particle_vars:
        src_lines.append('''
    /*gpufun*/
    void LocalParticle_add_to_'''+vv+f'(LocalParticle* part, {tt._c_type} value)'
    +'{')
        src_lines.append(f'  part->{vv}[part->ipart] += value;')
        src_lines.append('}\n')
    src_adders = '\n'.join(src_lines)

    src_lines=[]
    for tt, vv in per_particle_vars:
        src_lines.append('''
    /*gpufun*/
    void LocalParticle_scale_'''+vv+f'(LocalParticle* part, {tt._c_type} value)'
    +'{')
        src_lines.append(f'  part->{vv}[part->ipart] *= value;')
        src_lines.append('}\n')
    src_scalers = '\n'.join(src_lines)

    src_lines=[]
    for tt, vv in per_particle_vars:
        src_lines.append('''
    /*gpufun*/
    void LocalParticle_set_'''+vv+f'(LocalParticle* part, {tt._c_type} value)'
    +'{')
        src_lines.append(f'  part->{vv}[part->ipart] = value;')
        src_lines.append('}')
    src_setters = '\n'.join(src_lines)

    src_lines=[]
    for tt, vv in scalar_vars:
        src_lines.append('/*gpufun*/')
        src_lines.append(f'{tt._c_type} LocalParticle_get_'+vv
                        + f'(LocalParticle* part)'
                        + '{')
        src_lines.append(f'  return part->{vv};')
        src_lines.append('}')
    for tt, vv in per_particle_vars:
        src_lines.append('/*gpufun*/')
        src_lines.append(f'{tt._c_type} LocalParticle_get_'+vv
                        + f'(LocalParticle* part)'
                        + '{')
        src_lines.append(f'  return part->{vv}[part->ipart];')
        src_lines.append('}')
    src_getters = '\n'.join(src_lines)

    custom_source='''
/*gpufun*/
double LocalParticle_get_energy0(LocalParticle* part){

    double const p0c = LocalParticle_get_p0c(part);
    double const m0  = LocalParticle_get_mass0(part);

    return sqrt( p0c * p0c + m0 * m0 );
}

/*gpufun*/
void LocalParticle_add_to_energy(LocalParticle* part, double delta_energy){

    double const beta0 = LocalParticle_get_beta0(part);
    double const delta_beta0 = LocalParticle_get_delta(part) * beta0;

    double const ptau_beta0 =
        delta_energy / LocalParticle_get_energy0(part) +
        sqrt( delta_beta0 * delta_beta0 + 2.0 * delta_beta0 * beta0
                + 1. ) - 1.;

    double const ptau   = ptau_beta0 / beta0;
    double const psigma = ptau / beta0;
    double const delta = sqrt( ptau * ptau + 2. * psigma + 1 ) - 1;

    double const one_plus_delta = delta + 1.;
    double const rvv = one_plus_delta / ( 1. + ptau_beta0 );

    LocalParticle_set_delta(part, delta );
    LocalParticle_set_psigma(part, psigma );
    LocalParticle_scale_zeta(part,
        rvv / LocalParticle_get_rvv(part));

    LocalParticle_set_rvv(part, rvv );
    LocalParticle_set_rpp(part, 1. / one_plus_delta );
}
'''

    source = '\n\n'.join([src_typedef, src_particles_to_local, src_adders,
                          src_getters, src_setters, src_scalers, custom_source])

    return source

def pysixtrack_particles_to_xtrack_dict(pysixtrack_particles):


    if hasattr(pysixtrack_particles, '__iter__'):
        num_particles = len(pysixtrack_particles)
        raise NotImplementedError
        # Something recursive
        return
    else:
        out = {}

        # Vectorized everything
        pyst_dict = pysixtrack_particles.to_dict()
        for kk, vv in pyst_dict.items():
            pyst_dict[kk] = np.atleast_1d(vv)
        pyst_part_vectorized = pysixtrack.Particles.from_dict(pyst_dict)


        lll = [len(vv) for kk, vv in pyst_dict.items() if hasattr(vv, '__len__')]
        lll = list(set(lll))
        assert len(set(lll) - {1}) <= 1
        num_particles = max(lll)
        out['num_particles'] = num_particles

    for tt, vv in scalar_vars:
        if vv == 'num_particles':
            continue
        val = getattr(pyst_part_vectorized, vv)
        assert np.allclose(val, val[0], rtol=1e-10, atol=1e-14)
        out[vv] = val[0]

    for tt, vv in per_particle_vars:

        if vv == 'weight'and not hasattr(pysixtrack_particles, 'weight'):
            out['weight'] = np.ones(int(self.num_particles), dtype=tt._dtype)
            continue

        if vv == 'mass_ratio':
            vv_pyst = 'mratio'
        elif vv == 'charge_ratio':
            vv_pyst = 'qratio'
        elif vv == 'particle_id':
            vv_pyst = 'partid'
        elif vv == 'at_element':
            vv_pyst = 'elemid'
        elif vv == 'at_turn':
            vv_pyst = 'turn'
        else:
            vv_pyst = vv

        val_pyst = getattr(pyst_part_vectorized, vv_pyst)

        if num_particles > 1 and len(val_pyst)==1:
            temp = np.zeros(int(self.num_particles), dtype=tt._dtype)
            temp += val_pyst[0]
            val_pyst = temp

        if type(val_pyst) != tt._dtype:
            val_pyst = tt._dtype(val_pyst)

        out[vv] = val_pyst
