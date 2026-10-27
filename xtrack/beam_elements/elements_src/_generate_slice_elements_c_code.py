
import os
import re


def _is_setting(line, name):
    """Line setting `name`: a labelled argument (/*name*/ value,) or a
    MagnetSettings field (magnet.name = value;)."""
    return f"/*{name}*/" in line or f"magnet.{name} " in line


def _end(line):
    """End of a setting line: ';' for a MagnetSettings field, else ','."""
    return ';' if 'magnet.' in line else ','


# MagnetSettings (magnets): the struct is filled by MagnetSettings_init_default
# and an element sets only the other fields; the fields it leaves are listed
# in a comment below the settings, kept up to date here for the slices.
_INIT_DEFAULT = 'MagnetSettings_init_default(&magnet);'
_KEPT_HEAD = '// Kept at default (MagnetSettings_init_default):'
_KEPT_ORDER = [
    'weight', 'length', 'order', 'inv_factorial_order', 'knl', 'ksl',
    'order_rel', 'knl_rel', 'ksl_rel', 'rel_ref_strength',
    'num_multipole_kicks', 'model', 'integrator', 'radiation_flag',
    'radiation_record', 'delta_taper', 'h', 'hxl', 'k0', 'k1', 'k2', 'k3',
    'k0s', 'k1s', 'k2s', 'k3s', 'ks', 'dks_ds', 'x0_solenoid', 'y0_solenoid',
    'rbend_model', 'rbend_compensate_sagitta', 'rbend_shift',
    'rbend_angle_diff', 'length_straight', 'body_active']


def _kept_comment(names, indent):
    """Comment lines listing the fields kept at their default."""
    names = [nn for nn in _KEPT_ORDER if nn in names]
    out, cur = [indent + _KEPT_HEAD], indent + '//    '
    for nn in names:
        if len(cur) + len(nn) + 2 > 79:
            out.append(cur.rstrip())
            cur = indent + '//    '
        cur += ' ' + nn + ','
    out.append(cur.rstrip(','))
    return out


def _field_statement(lines, name):
    """Lines [i0, i1) of the statement setting a field (it may span several
    lines), or None."""
    for i0, line in enumerate(lines):
        if re.match(rf'\s*magnet\.{name}\s*=', line):
            i1 = i0
            while not re.sub(r'//.*', '', lines[i1]).rstrip().endswith(';'):
                i1 += 1
            return i0, i1 + 1
    return None


def _set_field(lines, name, value, comment=None):
    """Set a MagnetSettings field: replace its statement, or add one after
    the init (the field then leaves the "Kept at default" list)."""
    text = f'magnet.{name:<24} = {value};' + (f'  // {comment}' if comment else '')
    span = _field_statement(lines, name)
    if span is not None:
        i0, i1 = span
        lines[i0:i1] = [lines[i0][:len(lines[i0]) - len(lines[i0].lstrip())] + text]
        return
    i = next(i for i, l in enumerate(lines) if l.strip() == _INIT_DEFAULT)
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    lines.insert(i + 1, indent + text)
    _update_kept(lines)


def _unset_field(lines, name):
    """Remove the line setting a field: it keeps its default."""
    span = _field_statement(lines, name)
    if span is not None:
        del lines[span[0]:span[1]]
    _update_kept(lines)


def _update_kept(lines):
    """Rewrite the "Kept at default" comment from the fields set."""
    i0 = next(i for i, l in enumerate(lines) if l.strip() == _KEPT_HEAD)
    i1 = i0 + 1
    while lines[i1].strip().startswith('//    '):
        i1 += 1
    set_names = {re.match(r'\s*magnet\.(\w+)', l).group(1) for l in lines
                 if re.match(r'\s*magnet\.\w+\s*=', l)}
    indent = lines[i0][:len(lines[i0]) - len(lines[i0].lstrip())]
    lines[i0:i1] = _kept_comment(
        [nn for nn in _KEPT_ORDER if nn not in set_names], indent)


def slice_settings(lines, generated_class, generating):
    """MagnetSettings of a slice, from those of its parent: the slice has its
    own radiation flag (10: that of the parent), a thick or thin slice tracks
    its weight of the body, a thin slice one kick, an entry or exit slice only
    its face."""
    data = generated_class + 'Data'
    _set_field(lines, 'radiation_flag',
               f'magnet_radiation_flag({data}_get_radiation_flag(el), '
               f'{data}_get__parent_radiation_flag(el))')
    # Not supported in slices: kept at the default (NULL). The parent sets it
    # in an #ifdef XTRACK_TPSA_TRACK block, removed whole
    for k, line in enumerate(lines):
        if line.strip() == '#ifdef XTRACK_TPSA_TRACK' and re.match(
                r'\s*magnet\.radiation_record\s*=', lines[k + 1]):
            k1 = next(j for j in range(k, len(lines))
                      if lines[j].strip() == '#endif')
            del lines[k:k1 + 1]
            break
    _unset_field(lines, 'radiation_record')
    if generating in ('thick_slice', 'thin_slice'):
        _set_field(lines, 'weight', f'{data}_get_weight(el)')
    if generating == 'thin_slice':
        _set_field(lines, 'num_multipole_kicks', '1')
        _set_field(lines, 'model', 'MODEL_KICK_ONLY')
        _set_field(lines, 'integrator', 'INTEGRATOR_UNIFORM')
    if generating in ('entry_slice', 'exit_slice'):
        for name in ('num_multipole_kicks', 'model', 'integrator'):
            _unset_field(lines, name)
        _set_field(lines, 'body_active', '0', 'faces only')


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
    ("UniformSolenoid", "uniform_solenoid.h"),
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
        out_content = out_content.replace(generated_data_class + '_len_', generated_data_class + '_len__parent_')



        if parent_class not in ['Cavity', 'CrabCavity']:
            assert generated_data_class + '_getp1_' in out_content
        out_content = out_content.replace(generated_data_class + '_getp1_', generated_data_class + '_getp1__parent_')

        # delta_taper must come from the slice and not from the parent
        if parent_class not in ['Cavity', 'CrabCavity']:
            assert "_get__parent_delta_taper" in out_content
        out_content = out_content.replace("_get__parent_delta_taper", "_get_delta_taper")

        out_lines = out_content.splitlines()

        # Disable the faces a slice does not have: their MagnetFace keeps
        # MagnetFace_empty, so the face does nothing
        keep_entry = generating == 'entry_slice'
        keep_exit = generating == 'exit_slice'
        new_lines = []
        for line in out_lines:
            stripped = line.strip()
            if ((stripped.startswith('EDGE_LOAD_') and '(entry_face,' in stripped
                    and not keep_entry)
                    or (stripped.startswith('EDGE_LOAD_') and '(exit_face,' in stripped
                    and not keep_exit)):
                continue
            new_lines.append(line)
        out_lines = new_lines

        if _INIT_DEFAULT in out_content:
            # Magnets: set or reset the fields a slice changes
            slice_settings(out_lines, generated_class, generating)
        else:
            # Handle "radiation_flag" and "radiation_flag_parent"
    
            if parent_class not in ['Cavity', 'CrabCavity']:
                # Identify the line setting radiation_flag
                i_radiation_flag_line = None
                for i, line in enumerate(out_lines):
                    if _is_setting(line, "radiation_flag"):
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
                out_lines[i_parent_line] = ln_parent_radiation_flag + generated_data_class + "_get__parent_radiation_flag(el)" + _end(ln_parent_radiation_flag)

            # pass weight
            done_weight = False
            for i, line in enumerate(out_lines):
                if "weight" in line:
                    assert '1' in line, "Expected '1' in weight line"
                    ll = line
                    ll = ll.split('1')[0]
                    if generating == "thick_slice" or generating == "thin_slice":
                        ll += generated_data_class + "_get_weight(el)" + _end(line)
                    else:
                        ll += '0.' + _end(line) + ' // unused'
                    out_lines[i] = ll
                    done_weight = True
            assert done_weight, "Did not find weight line to modify"

            found_integrator = False
            found_model = False
            found_num_multipole_kicks = False
            found_enable_body = False
            found_radiation_record = False
            for i, line in enumerate(out_lines):
                if _is_setting(line, "integrator"):
                    found_integrator = True
                    ll = line
                    assert generated_data_class + '_get_' in ll
                    ll = ll.split(generated_data_class + '_get_')[0]
                    if generating == "thick_slice":
                        continue
                    elif generating == "thin_slice":
                        ll += "3" + _end(line) + " // uniform"
                    elif generating == "entry_slice" or generating == "exit_slice":
                        ll += "0" + _end(line) + " // unused"
                    else:
                        raise ValueError(f"Unknown generating type: {generating}")
                    out_lines[i] = ll
                if _is_setting(line, "model"):
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
                        ll += "-1" + _end(line) + " // kick only"
                    elif generating == "entry_slice" or generating == "exit_slice":
                        ll += "0" + _end(line) + " // unused"
                    else:
                        raise ValueError(f"Unknown generating type: {generating}")
                    out_lines[i] = ll
                if _is_setting(line, "num_multipole_kicks"):
                    found_num_multipole_kicks = True
                    ll = line
                    assert generated_data_class + '_get_' in ll
                    if generating == "thick_slice":
                        continue
                    elif generating == "thin_slice":
                        ll = ll.split(generated_data_class + '_get_')[0]
                        ll += "1" + _end(line) + " // kick only"
                    elif generating == "entry_slice" or generating == "exit_slice":
                        ll = ll.split(generated_data_class + '_get_')[0]
                        ll += "0" + _end(line) + " // unused"
                    else:
                        raise ValueError(f"Unknown generating type: {generating}")
                    out_lines[i] = ll
                if _is_setting(line, "num_kicks"):
                    found_num_kicks = True
                    ll = line
                    assert generated_data_class + '_get_' in ll
                    if generating == "thick_slice":
                        continue
                    elif generating == "thin_slice":
                        ll = ll.split(generated_data_class + '_get_')[0]
                        ll += "1" + _end(line) + " // kick only"
                    elif generating == "entry_slice" or generating == "exit_slice":
                        ll = ll.split(generated_data_class + '_get_')[0]
                        ll += "0" + _end(line) + " // unused"
                    else:
                        raise ValueError(f"Unknown generating type: {generating}")
                    out_lines[i] = ll
                if _is_setting(line, "body_active"):
                    found_enable_body = True
                    ll = line
                    assert '1' in ll
                    if generating == "thick_slice" or generating == "thin_slice":
                        continue
                    elif generating == "entry_slice" or generating == "exit_slice":
                        ll = ll.split('1')[0]
                        ll += "0" + _end(line) + " // disabled"
                    out_lines[i] = ll
                if _is_setting(line, "radiation_record"):
                    found_radiation_record = True
                    ll = line
                    if '_getp_' in ll:
                        assert '(SynchrotronRadiationRecordData)' in ll
                        ll = ll.split('(SynchrotronRadiationRecordData)')[0]
                        ll += 'NULL' + _end(line) # Not yet supported in slices
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

        out_content = "\n".join(out_lines) + "\n"

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

    #include "xtrack/headers/track.h"
    #include "xtrack/beam_elements/elements_src/track_drift.h"


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

    if parent_class == "RBend":

        out_drift_slice = '''
        // copyright ############################### //
        // This file is part of the Xtrack Package.  //
        // Copyright (c) CERN, 2023.                 //
        // ######################################### //

        #ifndef XTRACK_DRIFT_SLICE_RBEND_H
        #define XTRACK_DRIFT_SLICE_RBEND_H

        #include "xtrack/headers/track.h"
        #include "xtrack/beam_elements/elements_src/track_drift.h"


        GPUFUN
        void DriftSliceRBend_track_local_particle(
                DriftSliceRBendData el,
                LocalParticle* part0
        ) {

            double weight = DriftSliceRBendData_get_weight(el);
            int64_t rbend_model = DriftSliceRBendData_get__parent_rbend_model(el);
            double length_drift;
            double ds_corr = 0;
            double full_length_curved = DriftSliceRBendData_get__parent_length(el);
            if (rbend_model == 2) { // straight-body
                double full_length_straight = DriftSliceRBendData_get__parent_length_straight(el);
                length_drift = full_length_straight * weight;
                ds_corr = (full_length_curved - full_length_straight) * weight;
            }
            else {
                length_drift = full_length_curved * weight;
            }


            if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
                length_drift *= -1;
                ds_corr *= -1;
            }

            if (rbend_model == 2) { // straight-body
                // Force exact drift
                START_PER_PARTICLE_BLOCK(part0, part);
                    Drift_single_particle_exact(part, length_drift);
                END_PER_PARTICLE_BLOCK;
                START_PER_PARTICLE_BLOCK(part0, part);
                    LocalParticle_add_to_s(part, ds_corr);
                    LocalParticle_add_to_zeta(part, ds_corr);
                END_PER_PARTICLE_BLOCK;
            }
            else {
                START_PER_PARTICLE_BLOCK(part0, part);
                    Drift_single_particle(part, length_drift);
                END_PER_PARTICLE_BLOCK;
            }
        }

        #endif
        
'''

    parent_class_snake = ''.join(['_' + c.lower() if c.isupper() else c for c in parent_class]).lstrip('_')
    parent_class_snake = parent_class_snake.replace("r_bend", "rbend")

    out_drift_slice_snake = 'drift_slice_' + parent_class_snake + '.h'
    out_drift_slice_path = os.path.join(current_directory, out_drift_slice_snake)
    with open(out_drift_slice_path, 'w') as f:
        f.write("// This file is generated by _generate_slice_elements_c_code.py\n")
        f.write("// Do not edit it directly.\n\n")
        f.write(out_drift_slice)