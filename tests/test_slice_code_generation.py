from pathlib import Path
import runpy
import shutil


def test_generated_slice_edge_models(tmp_path):
    source_dir = (Path(__file__).parent.parent / 'xtrack' / 'beam_elements'
                  / 'elements_src')
    generator = '_generate_slice_elements_c_code.py'
    shutil.copy2(source_dir / generator, tmp_path / generator)
    for header in source_dir.glob('*.h'):
        shutil.copy2(header, tmp_path / header.name)
    runpy.run_path(str(tmp_path / generator))

    for parent, active_model in [('quadrupole', 1), ('uniform_solenoid', 3)]:
        for filename, expected_models in [
            (f'thick_slice_{parent}.h', (0, 0)),
            (f'thin_slice_{parent}_entry.h', (active_model, 0)),
            (f'thin_slice_{parent}_exit.h', (0, active_model)),
        ]:
            content = (tmp_path / filename).read_text()
            for side, expected_model in zip(('entry', 'exit'), expected_models):
                label = f'/*edge_{side}_model*/'
                model_lines = [line for line in content.splitlines() if label in line]
                assert len(model_lines) == 1, (filename, side)
                argument = model_lines[0].split(label, 1)[1]
                model = int(argument.split(',', 1)[0].strip())
                assert model == expected_model, (filename, side)
