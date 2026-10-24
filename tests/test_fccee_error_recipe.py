"""Physical magnet grouping must not create opposing trims inside a sextupole."""

from pathlib import Path
import importlib

import numpy as np
import pytest
import xtrack as xt


@pytest.fixture
def recipe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]
                                   / 'examples' / 'fccee_tuning'))
    return importlib.import_module('error_recipe')


def split_sextupole():
    elements = {
        'sfx1r.0': xt.Sextupole(length=.15, k2=.4),
        'sfx1r.1': xt.Sextupole(length=.15, k2=.4),
        'ocx.0': xt.Multipole(knl=[0, 0, 0, .02]),
        'decx.0': xt.Multipole(knl=[0, 0, 0, 0, .03]),
        'marker': xt.Marker(),
        'sfx1r.2': xt.Sextupole(length=.15, k2=.4),
        'sfx1r.3': xt.Sextupole(length=.15, k2=.4),
    }
    return xt.Line(elements=elements, element_names=list(elements))


def test_sextupole_halves_share_errors_across_thin_inserts(recipe, monkeypatch):
    line = split_sextupole()
    _, groups = recipe._inventory(line)
    assert [g['names'] for g in groups] == [
        ['sfx1r.0', 'sfx1r.1', 'sfx1r.2', 'sfx1r.3'], ['ocx.0'], ['decx.0']]
    assert groups[0]['end'] - groups[0]['s'] == pytest.approx(.6)
    assert groups[0]['centre'] == pytest.approx(.3)

    # This small line has no FCC-ee straight/arc boundary markers.
    def classify(tt, groups):
        for group in groups:
            group.update(category='straight_other', arc_index=0)
    monkeypatch.setattr(recipe, '_classify', classify)
    metadata = recipe.install_errors(line)
    assert metadata['magnets']['sfx1r.0']['names'] == groups[0]['names']
    assert 'sfx1r.2' not in metadata['magnets']
    for key in recipe.AXES:
        values = [getattr(line[n], key) for n in groups[0]['names']]
        np.testing.assert_array_equal(values, np.repeat(values[0], 4))
    errors = [line[n].knl_rel[2] for n in groups[0]['names']]
    np.testing.assert_array_equal(errors, np.repeat(errors[0], 4))
    assert errors[0] != 0.
    assert line['ocx.0'].main_order == 3
    assert line['decx.0'].main_order == 4
    line['on_errors'] = 0.
    for name in groups[0]['names']:
        assert line[name].shift_y == 0.
        assert line[name].knl_rel[2] == 0.


@pytest.mark.parametrize('barrier', ['gap', 'magnet', 'family', 'strength'])
def test_distinct_magnets_are_not_merged(recipe, barrier):
    elements = {'sd.0': xt.Sextupole(length=.3, k2=-.4),
                'oc.0': xt.Multipole(knl=[0, 0, 0, 1.])}
    if barrier == 'gap':
        elements['gap'] = xt.Drift(length=.01)
    elif barrier == 'magnet':
        elements['q.0'] = xt.Quadrupole(length=.1, k1=.2)
    name = 'other.0' if barrier == 'family' else 'sd.1'
    elements[name] = xt.Sextupole(length=.3, k2=-.5 if barrier == 'strength' else -.4)
    line = xt.Line(elements=elements, element_names=list(elements))
    _, groups = recipe._inventory(line)
    sextupoles = [g['names'] for g in groups if g['kind'] == 'Sextupole']
    assert sextupoles == [['sd.0'], [name]]
