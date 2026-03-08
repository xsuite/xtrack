import xtrack as xt
import xobjects as xo

line = xt.load('../../test_data/sps_w_spacecharge/line_no_spacecharge.json')

cls = xt.Multipole

src = '''
GPUKERN
void Multipole_get(
    /*gpuglmem*/ int8_t* buffer,
    /*gpuglmem*/ int64_t* offsets,
                 int64_t num_offsets,
                 int64_t field_id,
                 int64_t idx_within_field,
    /*gpuglmem*/ double* out_values){

    for (int64_t ii=0; ii<num_offsets; ii++) {
        int64_t ofst = offsets[ii];

        MultipoleData* el = (MultipoleData*)(buffer + ofst);

        if (field_id == 0) {
            int64_t len_field = MultipoleData_len_knl(*el);
            if (idx_within_field >= len_field) {
                out_values[ii] = 0;
            } else {
                out_values[ii] = MultipoleData_get_knl(*el, idx_within_field);
            }
        }
    }
}
'''

kernel_descriptions = {
    "Multipole_get": xo.Kernel(
        c_name='Multipole_get',
        args=[
            xo.Arg(xo.Int8, pointer=True, name="buffer"),
            xo.Arg(xo.Int64, pointer=True, name="offsets"),
            xo.Arg(xo.Int64, name="num_offsets"),
            xo.Arg(xo.Int64, name="field_id"),
            xo.Arg(xo.Int64, name="idx_within_field"),
            xo.Arg(xo.Float64, pointer=True, name="out_values"),
        ],
    )
}

context = xo.context_default
out_kernels = context.build_kernels(
    sources=[src],
    kernel_descriptions=kernel_descriptions,
    extra_headers=[],
    extra_classes=[xt.Multipole._XoStruct],
    apply_to_source=[],
    specialize=True,
    compile=True,
    save_source_as='src_kernels.c',
    extra_compile_args=(),
)



