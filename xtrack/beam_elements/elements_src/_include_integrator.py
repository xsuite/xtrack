
integration_code = \
'''
    // START GENERATED INTEGRATION CODE

    if (INTEGRATOR == 1){ // TEAPOT

        RADIATION_MACRO(LENGTH,
            const double kick_weight = 1. / NUM_KICKS;
            double edge_drift_weight = 0.5;
            double inside_drift_weight = 0;
            if (NUM_KICKS > 1) {
                edge_drift_weight = 1. / (2 * (1 + NUM_KICKS));
                inside_drift_weight = (
                    ((double) NUM_KICKS)
                        / ((double)(NUM_KICKS*NUM_KICKS) - 1));
            }

            DRIFT_FUNCTION(PART, edge_drift_weight*LENGTH);
            for (int i_kick=0; i_kick<NUM_KICKS - 1; i_kick++) {
                KICK_FUNCTION(PART, kick_weight);
                DRIFT_FUNCTION(PART, inside_drift_weight*LENGTH);
            }
            KICK_FUNCTION(PART, kick_weight);
            DRIFT_FUNCTION(PART, edge_drift_weight*LENGTH);
        )

    }
    else if (INTEGRATOR==3){ // uniform

        const double kick_weight = 1. / NUM_KICKS;
        const double drift_weight = kick_weight;

        for (int i_kick=0; i_kick<NUM_KICKS; i_kick++) {
            RADIATION_MACRO(drift_weight*LENGTH,
                DRIFT_FUNCTION(PART, 0.5*drift_weight*LENGTH);
                KICK_FUNCTION(PART, kick_weight);
                DRIFT_FUNCTION(PART, 0.5*drift_weight*LENGTH);
            )
        }

    }
    else if (INTEGRATOR==2){ // YOSHIDA 4

        const int64_t n_kicks_yoshida = 7;
        const int64_t num_slices = (NUM_KICKS / n_kicks_yoshida
                                + (NUM_KICKS % n_kicks_yoshida != 0));

        const double slice_LENGTH = LENGTH / (num_slices);
        const double kick_weight = 1. / num_slices;
        const double d_yoshida[] =
                     // From MAD-NG
                     {3.922568052387799819591407413100e-01,
                      5.100434119184584780271052295575e-01,
                      -4.710533854097565531482416645304e-01,
                      6.875316825251809316199569366290e-02};
                    //  {0x1.91abc4988937bp-2, 0x1.052468fb75c74p-1, // same in hex
                    //  -0x1.e25bd194051b9p-2, 0x1.199cec1241558p-4 };
                    //  {1/8.0, 1/8.0, 1/8.0, 1/8.0}; // Uniform, for debugging
        const double k_yoshida[] =
                     // From MAD-NG
                     {7.845136104775599639182814826199e-01,
                      2.355732133593569921359289764951e-01,
                      -1.177679984178870098432412305556e+00,
                      1.315186320683906284756403692882e+00};
                    //  {0x1.91abc4988937bp-1, 0x1.e2743579895b4p-3, // same in hex
                    //  -0x1.2d7c6f7933b93p+0, 0x1.50b00cfb7be3ep+0 };
                    //  {1/7.0, 1/7.0, 1/7.0, 1/7.0}; // Uniform, for debugging

            for (int ii = 0; ii < num_slices; ii++) {
                RADIATION_MACRO(slice_LENGTH,
                    DRIFT_FUNCTION(PART, slice_LENGTH * d_yoshida[0]);
                    KICK_FUNCTION(PART, kick_weight * k_yoshida[0]);
                    DRIFT_FUNCTION(PART, slice_LENGTH * d_yoshida[1]);
                    KICK_FUNCTION(PART, kick_weight * k_yoshida[1]);
                    DRIFT_FUNCTION(PART, slice_LENGTH * d_yoshida[2]);
                    KICK_FUNCTION(PART, kick_weight * k_yoshida[2]);
                    DRIFT_FUNCTION(PART, slice_LENGTH * d_yoshida[3]);
                    KICK_FUNCTION(PART, kick_weight * k_yoshida[3]);
                    DRIFT_FUNCTION(PART, slice_LENGTH * d_yoshida[3]);
                    KICK_FUNCTION(PART, kick_weight * k_yoshida[2]);
                    DRIFT_FUNCTION(PART, slice_LENGTH * d_yoshida[2]);
                    KICK_FUNCTION(PART, kick_weight * k_yoshida[1]);
                    DRIFT_FUNCTION(PART, slice_LENGTH * d_yoshida[1]);
                    KICK_FUNCTION(PART, kick_weight * k_yoshida[0]);
                    DRIFT_FUNCTION(PART, slice_LENGTH * d_yoshida[0]);
                ) // RADIATION_MACRO
            }
    } // INTEGRATOR if

    // END GENERATED INTEGRATION CODE
'''

fnames = [
    'track_magnet',
    'track_rf',
]

for ff in fnames:

    fname_in = f'{ff}.template.h'
    fname_out = f'{ff}.h'

    # In the template file search for something like:
    # INTEGRATION_CODE[[
    #   INTEGRATOR=myintegrator,
    #   DRIFT_FUNCTION=mydrift_function,
    #   KICK_FUNCTION=mykick_function,
    #   RADIATION_MACRO=radiation_macro,
    #   PART=mypart,
    #   LENGTH=mylength,
    #   NUM_KICKS=mynum_kicks
    # ]]
    # and replace it with the adapted code above.
    with open(fname_in, 'r') as f:
        content = f.read()

    # Find INTEGRATION_CODE and replace it
    start = content.find('INTEGRATION_CODE[[')
    end = content.find(']]', start)


    part_before = content[:start]
    part_integration_code = content[start:end + 2]  # include the closing brackets
    part_after = content[end + 2:]

    # Build a dictionary for the replacements by parsing the block (strip the spaces and the new lines,
    # split by commas and then split by the equal sign)
    part_replacements = part_integration_code.split('[[')[1].split(']]')[0].strip().split(',')
    replacements = {}
    for line in part_replacements:
        key, value = line.split('=')
        replacements[key.strip()] = value.strip()

    # Replace the INTEGRATION_CODE block with the new code
    new_integration_code = integration_code
    for key, value in replacements.items():
        new_integration_code = new_integration_code.replace(key, value)


    new_content = part_before + new_integration_code + part_after

    with open(fname_out, 'w') as f:
        f.write(new_content)