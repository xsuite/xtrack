import xtrack as xt

collider = xt.Multiline.from_json('hllhc15_collider_thick.json')

# dct = xt.json.load('hllhc15_collider_thick.json')

# lines = {}

# for line_name in dct['lines']:

#     dct_line = dct['lines'][line_name].copy()

#     new_man_data = []
#     for ee in dct['_var_manager']:
#         new_ee = []
#         skip = False
#         for cc in ee:
#             if 'eref' in cc and f"eref['{line_name}']" not in cc:
#                 skip = True
#                 break
#             new_cc = cc.replace(f"eref['{line_name}']", 'element_refs')
#             new_ee.append(new_cc)

#         if skip:
#             continue

#         new_man_data.append(tuple(new_ee))

#     dct_line['_var_management_data'] = {}
#     dct_line['_var_management_data']['var_values'] = dct['_var_management_data']['var_values'].copy()
#     dct_line['_var_manager'] = new_man_data

#     line = xt.Line.from_dict(dct_line)

#     lines[line_name] = line

# env = xt.Environment(lines=lines)
# env.metadata.update(dct['metadata'])
