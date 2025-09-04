
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

to_do = [
    ("RBend", "rbend.h"),
    ("Bend", "bend.h"),
    ("Octupole", "octupole.h"),
    ("Quadrupole", "quadrupole.h"),
    ("Sextupole", "sextupole.h"),
    ("UniformSolenoid", "slnd.h"),
    ("Cavity", "cavity.h"),
    ("CrabCavity", "crab_cavity.h"),
    ("Multipole", "multipole.h")
]

no_edge_for = ["Cavity", "CrabCavity", "Multipole"]

for td in to_do:

    parent_class = td[0]
    parent_source = td[1]

    # Read parent file
    with open(os.path.join(current_directory, parent_source), 'r') as f:
        parent_content = f.read()


    for generating in ['thick_slice', 'thin_slice', 'entry_slice', 'exit_slice']:

        if parent_class == "UniformSolenoid" and generating == "thin_slice":
            continue # Does not exist

        if parent_class in no_edge_for and generating == "entry_slice":
            continue # Does not exist

        if parent_class in no_edge_for and generating == "exit_slice":
            continue # Does not exist

        out_content = parent_content

        # Handle guards
        capitalized_parent_class = parent_class.upper()
        if capitalized_parent_class == 'UNIFORMSOLENOID':
            capitalized_parent_class = 'UNIFORM_SOLENOID'
        assert f"XTRACK_{capitalized_parent_class}_H" in out_content
        out_content = out_content.replace(f"XTRACK_{capitalized_parent_class}_H",
                    f"XTRACK_{generating.upper()}_{capitalized_parent_class}_H")

        if generating == "thick_slice":
            generated_class = "ThickSlice" + parent_class
        elif generating == "thin_slice":
            generated_class = "ThinSlice" + parent_class
        elif generating == "entry_slice":
            generated_class = "ThinSlice" + parent_class + "Entry"
        elif generating == "exit_slice":
            generated_class = "ThinSlice" + parent_class + "Exit"
        else:
            raise ValueError(f"Unknown generating type: {generating}")

        # Replace class name
        assert parent_class in out_content
        out_content = out_content.replace(parent_class, generated_class)

        generated_data_class = generated_class + "Data"

        assert generated_data_class + '_get_' in out_content
        out_content = out_content.replace(generated_data_class + '_get_', generated_data_class + '_get__parent_')
        out_content = out_content.replace(generated_data_class + '_getp_', generated_data_class + '_getp__parent_')

        if parent_class not in ['Cavity', 'CrabCavity']:
            assert generated_data_class + '_getp1_' in out_content
        out_content = out_content.replace(generated_data_class + '_getp1_', generated_data_class + '_getp1__parent_')

        # delta_taper must come from the slice and not from the parent
        if parent_class not in ['Cavity', 'CrabCavity']:
            assert "_get__parent_delta_taper" in out_content
        out_content = out_content.replace("_get__parent_delta_taper", "_get_delta_taper")

        # Handle "radiation_flag" and "radiation_flag_parent"
        out_lines = out_content.splitlines()

        if parent_class not in ['Cavity', 'CrabCavity']:
            # Identify the line with "/*radiation_flag*/"
            i_radiation_flag_line = None
            for i, line in enumerate(out_lines):
                if "/*radiation_flag*/" in line:
                    i_radiation_flag_line = i
                    break
            assert i_radiation_flag_line is not None, "Could not find radiation_flag line"
            i_parent_line = i_radiation_flag_line + 1

            assert "get__parent_radiation_flag" in out_lines[i_radiation_flag_line]
            out_lines[i_radiation_flag_line] = out_lines[i_radiation_flag_line].replace(
                "get__parent_radiation_flag", "get_radiation_flag")

            ln_parent_radiation_flag = out_lines[i_parent_line]
            assert "0" in ln_parent_radiation_flag, "Expected '0' in radiation_flag line"
            ln_parent_radiation_flag = ln_parent_radiation_flag.split('0')[0]
            out_lines[i_parent_line] = ln_parent_radiation_flag + generated_data_class + "_get__parent_radiation_flag(el),"

        # Disable edges where needed
        for i, line in enumerate(out_lines):
            if "edge_entry" in line or "edge_exit" in line:
                if generating == 'entry_slice' and 'edge_entry' in line:
                    continue # leave untouched
                if generating == 'exit_slice' and 'edge_exit' in line:
                    continue
                if '0' not in line:
                    if generated_data_class + '_get_' in line:
                        ll = line
                        ll = ll.split(generated_data_class + '_get_')[0]
                        ll += '0,'  # Disable edges by setting to 0
                    elif '1' in line or '0' in line or '3' in line:
                        ll = line
                        ll = ll.split('1')[0]
                        ll += '0,'
                    else:
                        raise ValueError(f"Unexpected line format: {line}")
                    if 'edge_exit_hgap' in ll:
                        # last argument, need to strip the comma
                        ll = ll.split(',')[0]
                    out_lines[i] = ll

        # pass weight
        done_weight = False
        for i, line in enumerate(out_lines):
            if "weight" in line:
                assert '1' in line, "Expected '1' in weight line"
                ll = line
                ll = ll.split('1')[0]
                if generating == "thick_slice" or generating == "thin_slice":
                    ll += generated_data_class + "_get_weight(el),"
                else:
                    ll += '0., // unused'
                out_lines[i] = ll
                done_weight = True
        assert done_weight, "Did not find weight line to modify"

        found_integrator = False
        found_model = False
        found_num_multipole_kicks = False
        found_enable_body = False
        found_radiation_record = False
        for i, line in enumerate(out_lines):
            if "/*integrator*/" in line:
                found_integrator = True
                ll = line
                assert generated_data_class + '_get_' in ll
                ll = ll.split(generated_data_class + '_get_')[0]
                if generating == "thick_slice":
                    continue
                elif generating == "thin_slice":
                    ll += "3, // uniform"
                elif generating == "entry_slice" or generating == "exit_slice":
                    ll += "0, // unused"
                else:
                    raise ValueError(f"Unknown generating type: {generating}")
                out_lines[i] = ll
            if "/*model*/" in line:
                found_model = True
                ll = line
                if generating == "thick_slice":
                    continue
                assert generated_data_class + '_get_' in ll or '-2' in ll # -2 used for solenoid
                if '((' in ll: #case of multipole that has some logic there
                    ll = ll.split('((')[0]
                elif generated_data_class + '_get_' in ll:
                    ll = ll.split(generated_data_class + '_get_')[0]
                elif '-2' in ll:
                    ll = ll.split('-2')[0]
                else:
                    raise ValueError
                if generating == "thin_slice":
                    ll += "-1, // kick only"
                elif generating == "entry_slice" or generating == "exit_slice":
                    ll += "0, // unused"
                else:
                    raise ValueError(f"Unknown generating type: {generating}")
                out_lines[i] = ll
            if "/*num_multipole_kicks*/" in line:
                found_num_multipole_kicks = True
                ll = line
                assert generated_data_class + '_get_' in ll
                if generating == "thick_slice":
                    continue
                elif generating == "thin_slice":
                    ll = ll.split(generated_data_class + '_get_')[0]
                    ll += "1, // kick only"
                elif generating == "entry_slice" or generating == "exit_slice":
                    ll = ll.split(generated_data_class + '_get_')[0]
                    ll += "0, // unused"
                else:
                    raise ValueError(f"Unknown generating type: {generating}")
                out_lines[i] = ll
            if "/*num_kicks*/" in line:
                found_num_kicks = True
                ll = line
                assert generated_data_class + '_get_' in ll
                if generating == "thick_slice":
                    continue
                elif generating == "thin_slice":
                    ll = ll.split(generated_data_class + '_get_')[0]
                    ll += "1, // kick only"
                elif generating == "entry_slice" or generating == "exit_slice":
                    ll = ll.split(generated_data_class + '_get_')[0]
                    ll += "0, // unused"
                else:
                    raise ValueError(f"Unknown generating type: {generating}")
                out_lines[i] = ll
            if "/*body_active*/" in line:
                found_enable_body = True
                ll = line
                assert '1' in ll
                if generating == "thick_slice" or generating == "thin_slice":
                    continue
                elif generating == "entry_slice" or generating == "exit_slice":
                    ll = ll.split('1')[0]
                    ll += "0, // disabled"
                out_lines[i] = ll
            if "/*radiation_record*/" in line:
                found_radiation_record = True
                ll = line
                if '_getp_' in ll:
                    assert '(SynchrotronRadiationRecordData)' in ll
                    ll = ll.split('(SynchrotronRadiationRecordData)')[0]
                    ll += 'NULL,' # Not yet supported in slices
                out_lines[i] = ll
        assert found_integrator, "Did not find integrator line to modify"
        assert found_model, "Did not find model line to modify"
        if parent_class in ["Cavity", "CrabCavity"]:
            assert found_num_kicks, "Did not find num_kicks line to modify"
        else:
            assert found_num_multipole_kicks, "Did not find num_multipole_kicks line to modify"
            assert found_radiation_record, "Did not find radiation_record line to modify"
        assert found_enable_body, "Did not find enable_body line to modify"

        # generated_class from camel case to snake case
        generated_class_snake = ''.join(['_' + c.lower() if c.isupper() else c for c in generated_class]).lstrip('_')

        # special case
        generated_class_snake = generated_class_snake.replace("_r_bend", "_rbend")

        out_content = "\n".join(out_lines)

        # Save
        out_file = os.path.join(current_directory, f"{generated_class_snake}.h")
        with open(out_file, 'w') as f:
            f.write("// This file is generated by _generate_slice_elements_c_code.py\n")
            f.write("// Do not edit it directly.\n\n")
            f.write(out_content)


    drift_slice_template = '''
    // copyright ############################### //
    // This file is part of the Xtrack Package.  //
    // Copyright (c) CERN, 2023.                 //
    // ######################################### //

    #ifndef XTRACK_DRIFT_SLICE_OCTUPOLE_H
    #define XTRACK_DRIFT_SLICE_OCTUPOLE_H

    #include <headers/track.h>
    #include <beam_elements/elements_src/track_drift.h>


    GPUFUN
    void DriftSliceOctupole_track_local_particle(
            DriftSliceOctupoleData el,
            LocalParticle* part0
    ) {

        double weight = DriftSliceOctupoleData_get_weight(el);
        double length;
        if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
            length = -weight * DriftSliceOctupoleData_get__parent_length(el); // m
        }
        else {
            length = weight * DriftSliceOctupoleData_get__parent_length(el); // m
        }

        START_PER_PARTICLE_BLOCK(part0, part);
            Drift_single_particle(part, length);
        END_PER_PARTICLE_BLOCK;
    }

    #endif
    '''
    out_drift_slice = drift_slice_template.replace("Octupole", parent_class)
    out_drift_slice = out_drift_slice.replace("Octupole".upper(), parent_class.upper())

    parent_class_snake = ''.join(['_' + c.lower() if c.isupper() else c for c in parent_class]).lstrip('_')
    parent_class_snake = parent_class_snake.replace("r_bend", "rbend")

    out_drift_slice_snake = 'drift_slice_' + parent_class_snake + '.h'
    out_drift_slice_path = os.path.join(current_directory, out_drift_slice_snake)
    with open(out_drift_slice_path, 'w') as f:
        f.write("// This file is generated by _generate_slice_elements_c_code.py\n")
        f.write("// Do not edit it directly.\n\n")
        f.write(out_drift_slice)