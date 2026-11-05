from pathlib import Path
import runpy
import shutil

import pytest


def test_generated_slice_edge_models(tmp_path):
    source_dir = (Path(__file__).parent.parent / 'xtrack' / 'beam_elements'
                  / 'elements_src')

    # In the case of the PyPI CI job, it keeps the tests but removes the
    # checkout's package directory so imports use the installed distribution.
    # The generator sources are then unavailable at this path, so this
    # source-generation test cannot run. We skip it in this case.
    if not source_dir.is_dir():
        pytest.skip('Slice code generation requires the source checkout')

    generator = '_generate_slice_elements_c_code.py'
    shutil.copy2(source_dir / generator, tmp_path / generator)
    for header in source_dir.glob('*.h'):
        shutil.copy2(header, tmp_path / header.name)
    runpy.run_path(str(tmp_path / generator))

    # A slice keeps the faces it has: the body slices none, an edge slice its
    # own face only (the others stay MagnetFace_empty, which does nothing),
    # and an edge slice has no body
    for parent in ('quadrupole', 'bend', 'uniform_solenoid'):
        for filename, faces, body in [
            (f'thick_slice_{parent}.h', (), True),
            (f'thin_slice_{parent}_entry.h', ('entry',), False),
            (f'thin_slice_{parent}_exit.h', ('exit',), False),
        ]:
            lines = (tmp_path / filename).read_text().splitlines()
            loaded = [side for side in ('entry', 'exit')
                      if any(ll.strip().startswith('EDGE_LOAD_')
                             and f'({side}_face,' in ll for ll in lines)]
            assert tuple(loaded) == faces, filename
            faces_only = any('magnet.body_active' in ll and '= 0' in ll for ll in lines)
            assert faces_only == (not body), filename
