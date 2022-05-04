import xobjects as xo
import xtrack as xt
import xpart as xp

class RecordIdentifier(xo.Struct):
    buffer_id = xo.Int64
    offset = xo.Int64
RecordIdentifier.extra_sources = []
RecordIdentifier.extra_sources.append('''

int8_t* RecordIdentifier_getp_record(RecordIdentifier record_id, LocalParticle* part){
    int8_t* io_buffer = LocalParticle_get_io_buffer(part);

    // TODO Check buffer_id
    // int64_t buffer_id = RecordIdentifier_get_buffer_id(record_id);

    int64_t offset = RecordIdentifier_get_offset(record_id);

    return io_buffer + offset;
    }

''')

class RecordIndex(xo.Struct):
    capacity = xo.Int64
    at_record = xo.Int64
    buffer_id = xo.Int64
RecordIndex.extra_sources = []
RecordIndex.extra_sources.append('''

int64_t RecordIndex_get_slot(RecordIndex record_index){

    int64_t capacity = RecordIndex_get_capacity(record_index);
    int64_t* at_record = RecordIndex_getp_at_record(record_index);

    if(*at_record >= capacity){
        return -1;}

    // TODO will have to be implemented with AtomicAdd, something like:
    // int64_t slot = AtomicAdd(at_record, 1);
    int64_t slot = *at_record;
    *at_record = slot + 1;

    return slot;
    }

''')

class TestElementRecord(xo.DressedStruct):
    _xofields = {
        '_record_index': RecordIndex,
        'generated_rr': xo.Float64[:],
        'at_element': xo.Int64[:]
        }

class TestElement(xt.BeamElement):
    _xofields={
        '_internal_record_id': RecordIdentifier,
        'n_iter': xo.Int64,
        }

    _skip_in_to_dict = ['_internal_record_id']

TestElement.XoStruct.extra_sources = [
    xp._pkg_root.joinpath('random_number_generator/rng_src/base_rng.h'),
    xp._pkg_root.joinpath('random_number_generator/rng_src/local_particle_rng.h'),
    RecordIdentifier._gen_c_api(),
    ]
TestElement.XoStruct.extra_sources.append(RecordIdentifier._gen_c_api())
TestElement.XoStruct.extra_sources += RecordIdentifier.extra_sources
TestElement.XoStruct.extra_sources.append(RecordIndex._gen_c_api())
TestElement.XoStruct.extra_sources += RecordIndex.extra_sources
TestElement.XoStruct.extra_sources.append(TestElementRecord.XoStruct._gen_c_api())

TestElement.XoStruct.extra_sources.append(r'''
    /*gpufun*/
    void TestElement_track_local_particle(TestElementData el, LocalParticle* part0){

        // Extract the record_id, record and record_index
        RecordIdentifier record_id = TestElementData_getp__internal_record_id(el);
        TestElementRecordData record =
           (TestElementRecordData) RecordIdentifier_getp_record(record_id, part0);
        RecordIndex record_index = NULL;
        if (record) record_index = TestElementRecordData_getp__record_index(record);

        int64_t n_iter = TestElementData_get_n_iter(el);
        printf("n_iter %d\n", (int)n_iter);

        //start_per_particle_block (part0->part)

            for (int64_t i = 0; i < n_iter; i++) {
                double rr = LocalParticle_generate_random_double(part);
                LocalParticle_add_to_x(part, rr);

                int64_t i_slot = RecordIndex_get_slot(record_index); // gives negative is record is NULL or if record is full
                printf("Hello %d\n", (int)i_slot);
                if (i_slot>=0){
                    TestElementRecordData_set_at_element(record, i_slot,
                                                LocalParticle_get_at_element(part));
                    TestElementRecordData_set_generated_rr(record, i_slot,
                                                LocalParticle_get_at_element(part));
                }
            }


        //end_per_particle_block
    }
    ''')

tracker = xt.Tracker(line=xt.Line(elements = [TestElement(n_iter=2)]))
tracker.line._needs_rng = True

# We could do something like
# tracker.start_internal_logging_for_elements_of_type(TestElement, num_records=10000)

part = xp.Particles(p0c=6.5e12, x=[1,2,3])

tracker.track(part)