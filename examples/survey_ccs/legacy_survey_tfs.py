import numpy as np


_HEADER = """
@ NAME             %06s "SURVEY"
@ TYPE             %06s "SURVEY"
@ TITLE            %08s "no-title"
@ ORIGIN           %17s "5.09.03 Darwin 64"
@ DATE             %08s "02/09/26"
@ TIME             %08s "12.10.29"
* NAME               KEYWORD                                    S                         L                     ANGLE                         X                         Y                         Z                     THETA                       PHI                       PSI                GLOBALTILT                      TILT    SLOT_ID ASSEMBLY_ID                  MECH_SEP                     V_POS
$ %s                 %s                                       %le                       %le                       %le                       %le                       %le                       %le                       %le                       %le                       %le                       %le                       %le         %d         %d                       %le                       %le
""".lstrip()


def write_legacy_survey_tfs(
        file_name, *, survey, element_names, element_container):
    """Write element entrance and exit frames in the legacy survey format."""
    lines = []
    for ii, nn in enumerate(element_names):
        frames = survey.get_all_frames(nn)
        for place in ('start', 'end'):

            if place == 'start':
                frame = frames['elem_start']
                name = f'drift_{ii}'
                s = survey['s', nn]
                length = 0
                slot_id = 0
            else:
                frame = frames['elem_end']
                name = nn
                s = survey['s', nn + '>>1']
                length = np.linalg.norm(
                    frames['elem_end'].XYZ - frames['elem_start'].XYZ, 2)
                slot_id = element_container[nn].extra.get('slot_id', 0)

            line = ' '

            # NAME
            value = f'"{name.upper()}"'.ljust(20)
            line += value

            # KEYWORD
            value = '"ELEMENT"'.ljust(20)
            line += value

            # S
            line += f'{s:26.9f}'

            # L
            line += f'{length:26.9f}'

            # ANGLE
            line += f'{0:26.9f}'

            # X, Y, Z, THETA, PHI, PSI
            for value in (
                    frame.X, frame.Y, frame.Z,
                    frame.theta, frame.phi, frame.psi):
                line += f'{value:26.9f}'

            # GLOBALTILT
            line += f'{frame.psi:26.9f}'

            # TILT
            line += f'{0:26.9f}'

            # SLOT_ID
            line += f'{int(slot_id):11d}'

            # ASSEMBLY_ID
            line += f'{0:11d}'

            # MECH_SEP
            line += f'{0:26.9f}'

            # V_POS
            line += f'{0:26.9f}'

            lines.append(line)

    output = _HEADER + '\n'.join(lines)
    with open(file_name, 'w') as fid:
        fid.write(output)
