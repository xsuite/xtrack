import pathlib

import xtrack as xt

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

collider = xt.Multiline.from_json(
    test_data_folder / 'hllhc15_thick/hllhc15_collider_thick.json')