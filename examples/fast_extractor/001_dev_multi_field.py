import xtrack as xt
import xobjects as xo

import numpy as np

line = xt.load('../../test_data/sps_w_spacecharge/line_no_spacecharge.json')

cls = xt.Multipole

name_to_id = {}
for ii, nn in enumerate(cls._xofields):
    name_to_id[nn] = ii

id_to_name = {ii: nn for nn, ii in name_to_id.items()}

src_start = '''
#ifndef XTRACK_!!CLSNAME!!_MULTI_GET
#define XTRACK_!!CLSNAME!!_MULTI_GET

GPUKERN
void !!CLSNAME!!_multi_get(
    /*gpuglmem*/ int8_t* buffer,
    /*gpuglmem*/ int64_t* offsets,
                 int64_t num_offsets,
                 int64_t field_id,
                 int64_t idx_within_field,
    /*gpuglmem*/ int8_t* out_ptr){

    VECTORIZE_OVER(ii, num_offsets);
        int64_t ofst = offsets[ii];

        /*gpuglmem*/ int8_t* el_pointer = buffer + ofst;
        !!CLSNAME!!Data el = (!!CLSNAME!!Data) el_pointer;

'''.replace('!!CLSNAME!!', cls.__name__)

SUPPORTED_CTYPES = {'int8_t', 'int64_t', 'double', 'int32_t'}

src_body_parts = []
for iidd in id_to_name:
    nn = id_to_name[iidd]
    typ = cls._xofields[nn]
    if isinstance(typ, xo.Field):
        typ = typ.ftype
    isarray = hasattr(typ, 'to_nparray')

    if isarray:
        ctype_name = cls._xofields['knl']._itemtype._c_type
    else:
        ctype_name = typ._c_type

    if ctype_name not in SUPPORTED_CTYPES:
        continue

    if isarray:
        src_part = '''
            case !!IIDD!!: {
                !!CCTTYP!!* out_values = (!!CCTTYP!!*) out_ptr;

                int64_t len_field = !!CLSNAME!!Data_len_!!FIELDNAME!!(el);
                if (idx_within_field >= len_field) {
                    out_values[ii] = 0;
                } else {
                    out_values[ii] = !!CLSNAME!!Data_get_!!FIELDNAME!!(el, idx_within_field);
                }
                break;
            }
        '''.replace('!!IIDD!!', str(iidd)).replace('!!CCTTYP!!', ctype_name).replace('!!CLSNAME!!', cls.__name__).replace('!!FIELDNAME!!', nn)
    else:
        src_part = '''
            case !!IIDD!!: {
                !!CCTTYP!!* out_values = (!!CCTTYP!!*) out_ptr;
                out_values[ii] = !!CLSNAME!!Data_get_!!FIELDNAME!!(el);
                break;
            }
        '''.replace('!!IIDD!!', str(iidd)).replace('!!CCTTYP!!', ctype_name).replace('!!CLSNAME!!', cls.__name__).replace('!!FIELDNAME!!', nn)

    src_body_parts.append(src_part)

src_end = '''
    END_VECTORIZE;
}
#endif
'''

src_switch_start = '''
        switch (field_id) {
'''

src_switch_end = '''
            default:
                break;
        }
'''

src = '\n'.join([src_start, src_switch_start] + src_body_parts + [src_switch_end, src_end])

kernel_descriptions = {
    cls.__name__ + "_multi_get": xo.Kernel(
        c_name=cls.__name__ + '_multi_get',
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
    extra_classes=[cls._XoStruct],
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

field_name = 'order'
typ = cls._xofields[field_name]
if isinstance(typ, xo.Field):
    typ = typ.ftype
isarray = hasattr(typ, 'to_nparray')
if isarray:
    dtype = cls._xofields[field_name]._itemtype._dtype
else:
    dtype = typ._dtype

out = np.zeros_like(mult_offsets, dtype=dtype)
kernel = out_kernels[cls.__name__ + "_multi_get"]
kernel(
    buffer=buffer.buffer,
    offsets=mult_offsets,
    num_offsets=len(mult_offsets),
    field_id=name_to_id[field_name],
    idx_within_field=1,
    out_values=out.view(np.int8)
)

