fname = 'aperturedb_classes.madx'

with open(fname, 'r') as fid:
    lines = fid.readlines()

# if the line ends with semicolon and contains apertype, prepend "SPS_" to the line

for ii, line in enumerate(lines):
    if 'apertype' in line.lower():
        lines[ii] = 'SPS_' + line

with open('apertures_old_model_new_naming.madx', 'w') as fid:
    fid.writelines(lines)

