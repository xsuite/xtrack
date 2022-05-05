import xobjects as xo
import xtrack as xt
import xpart as xp

from xtrack import RecordIdentifier, RecordIndex

class TestElementRecord(xo.DressedStruct):
    _xofields = {
        '_record_index': RecordIndex,
        'generated_rr': xo.Float64[:],
        'at_element': xo.Int64[:],
        'at_turn': xo.Int64[:]
        }

class TestElement(xt.BeamElement):
    _xofields={
        '_internal_record_id': RecordIdentifier,
        'n_iter': xo.Int64,
        }

    _skip_in_to_dict = ['_internal_record_id']


TestElement.XoStruct.internal_record_class = TestElementRecord

TestElement.XoStruct.extra_sources = [
    xp._pkg_root.joinpath('random_number_generator/rng_src/base_rng.h'),
    xp._pkg_root.joinpath('random_number_generator/rng_src/local_particle_rng.h'),
    ]

TestElement.XoStruct.extra_sources.append(r'''
    /*gpufun*/
    void TestElement_track_local_particle(TestElementData el, LocalParticle* part0){

        // Check if internal record is enabled
        int record_enabled = TestElementData_get__internal_record_id_buffer_id(el) > 0;

        TestElementRecordData record = NULL;
        RecordIndex record_index = NULL;

        // Extract the record_id, record and record_index
        if (record_enabled){
            RecordIdentifier record_id = TestElementData_getp__internal_record_id(el);
            record = (TestElementRecordData) RecordIdentifier_getp_record(record_id, part0);
            if (record){
                record_index = TestElementRecordData_getp__record_index(record);
            }
        }

        int64_t n_iter = TestElementData_get_n_iter(el);
        printf("n_iter %d\n", (int)n_iter);

        //start_per_particle_block (part0->part)

            for (int64_t i = 0; i < n_iter; i++) {
                double rr = LocalParticle_generate_random_double(part);
                LocalParticle_add_to_x(part, rr);

                if (record_enabled){
                    int64_t i_slot = RecordIndex_get_slot(record_index);
                    // gives negative is record is NULL or if record is full

                    printf("Hello %d\n", (int)i_slot);
                    if (i_slot>=0){
                        TestElementRecordData_set_at_element(record, i_slot,
                                                    LocalParticle_get_at_element(part));
                        TestElementRecordData_set_generated_rr(record, i_slot, rr);
                    }
                }
            }


        //end_per_particle_block
    }
    ''')



def start_internal_logging_for_elements_of_type(tracker, element_type, capacity):

    init_capacities = {}
    for ff in element_type.XoStruct.internal_record_class.XoStruct._fields:
        if hasattr(ff.ftype, 'to_nplike'): #is array
            init_capacities[ff.name] = capacity

    record = element_type.XoStruct.internal_record_class(_buffer=tracker.io_buffer, **init_capacities)
    record._record_index.capacity = capacity

    for ee in tracker.line.elements:
        if isinstance(ee, element_type):
            ee._internal_record_id.offset = record._offset
            ee._internal_record_id.buffer_id = xo.Int64._from_buffer(
                                                            record._buffer, 0)

    return record

tracker = xt.Tracker(line=xt.Line(elements = [TestElement(n_iter=2)]))
tracker.line._needs_rng = True

# We could do something like
record = start_internal_logging_for_elements_of_type(tracker, TestElement, capacity=10000)

part = xp.Particles(p0c=6.5e12, x=[1,2,3])

tracker.track(part, num_turns=10)