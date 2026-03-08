import xtrack as xt
import xobjects as xo

import numpy as np

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
    /*gpuglmem*/ int8_t* out_ptr){

    VECTORIZE_OVER(ii, num_offsets);
        int64_t ofst = offsets[ii];

        /*gpuglmem*/ int8_t* el_pointer = buffer + ofst;
        MultipoleData el = (MultipoleData) el_pointer;

        if (field_id == 0) {
            double* out_values = (double*) out_ptr;

            int64_t len_field = MultipoleData_len_knl(el);
            if (idx_within_field >= len_field) {
                out_values[ii] = 0;
            } else {
                out_values[ii] = MultipoleData_get_knl(el, idx_within_field);
            }
        }
    END_VECTORIZE;
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
            xo.Arg(xo.Int8, pointer=True, name="out_values"),
        ],
        n_threads='num_offsets',
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

# Test the kernel
line.build_tracker()
buffer = line._buffer

tt = line.get_table()
mult_names = []
mult_offsets = []
for nn in tt.name:
    if nn == '_end_point':
        continue
    if isinstance(line[nn], xt.Multipole):
        assert line[nn]._buffer is buffer, f"Buffer mismatch for element {nn}"
        mult_names.append(nn)
        mult_offsets.append(line[nn]._offset)

mult_offsets = np.array(mult_offsets, dtype=np.int64)

out = np.zeros_like(mult_offsets, dtype=np.float64)
kernel = out_kernels['Multipole_get']
kernel(
    buffer=buffer.buffer,
    offsets=mult_offsets,
    num_offsets=len(mult_offsets),
    field_id=0,
    idx_within_field=1,
    out_values=out.view(np.int8)
)


