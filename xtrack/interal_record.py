import numpy as np
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
/*gpuglmem*/ int8_t* RecordIdentifier_getp_record(RecordIdentifier record_id, LocalParticle* part){
    /*gpuglmem*/ int8_t* io_buffer = LocalParticle_get_io_buffer(part);
    if (io_buffer == NULL){
        return NULL;
    }

    int64_t buffer_id = RecordIdentifier_get_buffer_id(record_id);
    /*gpuglmem*/ int64_t* found_id = (/*gpuglmem*/ int64_t*)io_buffer;
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
    num_recorded = xo.UInt32
    _dummy = xo.UInt32 # to make sure the size is a multiple of 64 bits (not really needed)
    buffer_id = xo.Int64
RecordIndex.extra_sources = []
RecordIndex.extra_sources.append('''

/*gpufun*/
int64_t RecordIndex_get_slot(RecordIndex record_index){

    if (record_index == NULL){
        return -2;
    }
    int64_t capacity = RecordIndex_get_capacity(record_index);
    /*gpuglmem*/ uint32_t* num_recorded = RecordIndex_getp_num_recorded(record_index);

    if(*num_recorded >= capacity){
        return -1;}

    uint32_t slot = atomic_add(num_recorded, 1);   //only_for_context opencl
    uint32_t slot = atomicAdd(num_recorded, 1);    //only_for_context cuda
    uint32_t slot = *num_recorded;                 //only_for_context cpu_serial
    *num_recorded = slot + 1;                      //only_for_context cpu_serial

    return (int64_t) slot;
    }

''')

def start_internal_logging_for_elements_of_type(tracker, element_type, capacity):

    init_dict = {}
    if np.isscalar(capacity):
        capacity = int(capacity)
        for ff in element_type._internal_record_class.XoStruct._fields:
            if hasattr(ff.ftype, 'to_nplike'): #is array
                init_dict[ff.name] = capacity
    else:
        init_dict = {}
        for ff in element_type._internal_record_class.XoStruct._fields:
            if ff.name in capacity.keys():
                subtable_class = ff.ftype
                init_dict[ff.name] = {}
                for sff in subtable_class._fields:
                    if hasattr(sff.ftype, 'to_nplike'): #is array
                        init_dict[ff.name][sff.name] = capacity[ff.name]
    record = element_type.XoStruct._internal_record_class(_buffer=tracker.io_buffer, **init_dict)

    if np.isscalar(capacity):
        record._index.capacity = capacity
    else:
        for kk in capacity.keys():
            getattr(record, kk)._index.capacity = capacity[kk]

    for ee in tracker.line.elements:
        if isinstance(ee, element_type):
            ee._internal_record_id.offset = record._offset
            ee._internal_record_id.buffer_id = xo.Int64._from_buffer(
                                                            record._buffer, 0)
            ee.io_buffer = tracker.io_buffer
            ee.record = record
    return record

def stop_internal_logging_for_elements_of_type(tracker, element_type):

    for ee in tracker.line.elements:
        if isinstance(ee, element_type):
            ee._internal_record_id.offset = 0
            ee._internal_record_id.buffer_id = 0
            ee.io_buffer = None

def generate_get_record(ele_classname, record_classname):
    content = '''
RECORDCLASSNAME ELECLASSNAME_getp_internal_record(ELECLASSNAME el, LocalParticle* part){
    RecordIdentifier record_id = ELECLASSNAME_getp__internal_record_id(el);
    if (RecordIdentifier_get_buffer_id(record_id) <= 0){
        return NULL;
    }
    else{
        return (RECORDCLASSNAME) RecordIdentifier_getp_record(record_id, part);
    }
    }
    '''.replace(
        'RECORDCLASSNAME', record_classname).replace('ELECLASSNAME', ele_classname)
    return content