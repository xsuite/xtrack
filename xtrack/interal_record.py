import xobjects as xo

class RecordIdentifier(xo.Struct):
    '''
    To be inserted in the beam element.
    '''
    buffer_id = xo.Int64
    offset = xo.Int64
RecordIdentifier.extra_sources = []
RecordIdentifier.extra_sources.append(r'''
/*gpufun*/
int8_t* RecordIdentifier_getp_record(RecordIdentifier record_id, LocalParticle* part){
    int8_t* io_buffer = LocalParticle_get_io_buffer(part);
    if (io_buffer == NULL){
        return NULL;
    }

    int64_t buffer_id = RecordIdentifier_get_buffer_id(record_id);
    int64_t* found_id = (int64_t*)io_buffer;
    if (buffer_id != (*found_id)){
        printf("Error: buffer_id mismatch!\n");
        return NULL;
    }

    int64_t offset = RecordIdentifier_get_offset(record_id);

    return io_buffer + offset;
    }

''')

class RecordIndex(xo.Struct):
    '''
    To be inserted in the record class.
    '''
    capacity = xo.Int64
    num_recorded = xo.Int32
    _dummy = xo.Int32 # to make sure the size is a multiple of 64 bits (not really needed)
    buffer_id = xo.Int64
RecordIndex.extra_sources = []
RecordIndex.extra_sources.append('''

/*gpufun*/
int64_t RecordIndex_get_slot(RecordIndex record_index){

    if (record_index == NULL){
        return -2;
    }
    int64_t capacity = RecordIndex_get_capacity(record_index);
    int32_t* num_recorded = RecordIndex_getp_num_recorded(record_index);

    if(*num_recorded >= capacity){
        return -1;}

    // TODO will have to be implemented with AtomicAdd, something like:
    int32_t slot = atomicInc(num_recorded);    //only_for_context cuda
    int32_t slot = *num_recorded;              //only_for_context cpu_serial
    *num_recorded = slot + 1;                  //only_for_context cpu_serial

    return (int64_t) slot;
    }

''')

def start_internal_logging_for_elements_of_type(tracker, element_type, capacity):

    init_capacities = {}
    for ff in element_type.XoStruct._internal_record_class.XoStruct._fields:
        if hasattr(ff.ftype, 'to_nplike'): #is array
            init_capacities[ff.name] = capacity

    record = element_type.XoStruct._internal_record_class(_buffer=tracker.io_buffer, **init_capacities)
    record._record_index.capacity = capacity

    for ee in tracker.line.elements:
        if isinstance(ee, element_type):
            ee._internal_record_id.offset = record._offset
            ee._internal_record_id.buffer_id = xo.Int64._from_buffer(
                                                            record._buffer, 0)
            ee.io_buffer = tracker.io_buffer
    return record

def stop_internal_logging_for_elements_of_type(tracker, element_type):

    for ee in tracker.line.elements:
        if isinstance(ee, element_type):
            ee._internal_record_id.offset = 0
            ee._internal_record_id.buffer_id = 0
            ee.io_buffer = None
