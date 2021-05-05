typedef signed long long int64_t;
typedef signed char      int8_t;

      __kernel 
     void track_line(
          __global int* test,
          __global  int8_t* buffer,
          __global  int64_t* ele_offsets,
          __global  int64_t* ele_typeids,
                      int64_t ele_start,
                      int64_t num_ele_track
		      ){
 
 
	int part_id = get_global_id(0);                    //only_for_context opencl
}
