# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import os

copyright_statement = (
"""copyright ###############################
This file is part of the Xtrack Package. 
Copyright (c) CERN, 2021.                
#########################################""")

config = [
    {'extension': '.py', 'comment_char': '#'},
    {'extension': '.h', 'comment_char': '//'},
]

for cc in config:
    extension =  cc['extension']
    comment_char = cc['comment_char']

    cpright_lines = [comment_char + ' ' + line + " " + comment_char + '\n'
                        for line in copyright_statement.splitlines()
                        ] + ['\n']

    for root, dirs, files in os.walk("./"):
        for fname in files:
            if fname.endswith(extension):
                file = os.path.join(root, fname)
                print(file)
                with open(file, 'r') as fid:
                    lines = fid.readlines()
                # Remove copyright statement if present
                if (len(lines) > 0 and
                    lines[0].startswith(comment_char + ' ' + 'copyright ##')):
                    for ill, ll in enumerate(lines):
                        assert ll.startswith(comment_char)
                        if ll.startswith(comment_char + ' ' + '########'):
                            end_cpright = ill
                            break
                    assert lines[end_cpright+1] == '\n'
                    lines = lines[end_cpright+2:]

                lines = cpright_lines + lines

                with open(file, 'w') as fid:
                    fid.writelines(lines)
