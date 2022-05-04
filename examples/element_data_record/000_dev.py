import xobjects as xo
import xtrack as xt
import xpart as xp

class RecordIdentifier(xo.Struct):
    buffer_id = xo.Int64
    offset = xo.Int64

class RecordIndex(xo.Struct):
    capacity = xo.Int64
    at_record = xo.Int64
    buffer_id = xo.Int64

class TestElementRecord(xo.DressedStruct):
    _xofields = {
        '_record_index': RecordIndex,
        'myarray1': xo.Float64[:],
        'myarray2': xo.Float64[:]
        }

class TestElement(xt.BeamElement):
    _xofields={
        '_internal_record_id': RecordIdentifier,
        'dummy': xo.Float64,
        }

    _skip_in_to_dict = ['_internal_record_id']

TestElement.XoStruct.extra_sources = [
    xp._pkg_root.joinpath('random_number_generator/rng_src/base_rng.h'),
    xp._pkg_root.joinpath('random_number_generator/rng_src/local_particle_rng.h')]

TestElement.XoStruct.extra_sources.append('''
    /*gpufun*/
    void TestElement_track_local_particle(
        TestElementData el, LocalParticle* part0){

        RecordIdentifier record_id= TestElementData_getp__internal_record_id(el);

        //start_per_particle_block (part0->part)
            double rr = LocalParticle_generate_random_double(part);
            LocalParticle_add_to_x(part, rr);


            rr = LocalParticle_generate_random_double(part);
            LocalParticle_add_to_x(part, rr);

            rr = LocalParticle_generate_random_double(part);
            LocalParticle_add_to_x(part, rr);

        //end_per_particle_block
    }
    ''')

tracker = xt.Tracker(line=xt.Line(elements = [TestElement()]))
tracker.line._needs_rng = True

# We could do something like
# tracker.start_internal_logging_for_elements_of_type(TestElement, num_records=10000)

part = xp.Particles(p0c=6.5e12, x=[1,2,3])

tracker.track(part)