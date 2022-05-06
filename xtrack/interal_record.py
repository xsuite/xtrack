import xobjects as xo

class RecordIdentifier(xo.Struct):
    '''
    To be inserted in the beam element.
    '''
    buffer_id = xo.Int64
    offset = xo.Int64
RecordIdentifier.extra_sources = []
RecordIdentifier.extra_sources.append(r'''
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
    num_recorded = xo.Int64
    buffer_id = xo.Int64
RecordIndex.extra_sources = []
RecordIndex.extra_sources.append('''

int64_t RecordIndex_get_slot(RecordIndex record_index){

    if (record_index == NULL){
        return -2;
    }
    int64_t capacity = RecordIndex_get_capacity(record_index);
    int64_t* num_recorded = RecordIndex_getp_num_recorded(record_index);

    if(*num_recorded >= capacity){
        return -1;}

    // TODO will have to be implemented with AtomicAdd, something like:
    // int64_t slot = AtomicAdd(num_recorded, 1);
    int64_t slot = *num_recorded;
    *num_recorded = slot + 1;

    return slot;
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
