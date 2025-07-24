
import os

current_directory = os.path.dirname(__file__)

# # list all files starting with "thick_slice_" or "thin_slice_"in the current directory
# files = os.listdir(current_directory)
# old_slice_files = [f for f in files if f.startswith("thick_slice")
#                    or f.startswith("thin_slice") or f.startswith("drift_slice_")]
# # Rename them adding "old_" prefix
# for old_file in old_slice_files:
#     new_file = "old_" + old_file
#     os.rename(os.path.join(current_directory, old_file), os.path.join(current_directory, new_file))

parent_class = "RBend"
parent_source = "rbend.h"

# Read parent file
with open(os.path.join(current_directory, parent_source), 'r') as f:
    parent_content = f.read()

generating = "thick_slice"

out_content = parent_content

# Handle guards
assert f"XTRACK_{parent_class.upper()}_H" in out_content
out_content = out_content.replace(f"XTRACK_{parent_class.upper()}_H",
                                  f"XTRACK_{generating.upper()}_{parent_class.upper()}_H")

if generating == "thick_slice":
    generated_class = "ThickSlice" + parent_class
else:
    raise ValueError(f"Unknown generating type: {generating}")

# Replace class name
assert parent_class in out_content
out_content = out_content.replace(parent_class, generated_class)

generated_data_class = generated_class + "Data"

assert generated_data_class + '_get_' in out_content
out_content = out_content.replace(generated_data_class + '_get_', generated_class + '_get_parent__')

assert generated_data_class + '_getp1_' in out_content
out_content = out_content.replace(generated_data_class + '_getp1_', generated_class + '_getp1_parent__')
