import pytest

import xtrack as xt


class PythonElement:
    iscollective = True
    isthick = True
    length = 2.


@pytest.mark.parametrize('container', ['line', 'env'])
@pytest.mark.parametrize('method', ['to_dict', 'to_json'])
@pytest.mark.parametrize('element_kind', ['python', 'noncallable', 'rft'])
def test_unsupported_element_serialization(
        container, method, element_kind, tmp_path):
    if element_kind == 'rft':
        rft = pytest.importorskip('RF_Track')
        element = xt.RFT_Element(rft.Drift(2.))
    else:
        element = PythonElement()
        if element_kind == 'noncallable':
            element.to_dict = None

    env = xt.Environment()
    env.elements['unsupported'] = element
    line = env.new_line(components=['unsupported'])
    obj = line if container == 'line' else env

    path = tmp_path / 'line.json'
    with pytest.raises(TypeError) as exc:
        if method == 'to_dict':
            obj.to_dict()
        else:
            obj.to_json(path)
    assert str(exc.value) == (
        f"Cannot serialize element 'unsupported' "
        f"(type {type(element).__name__}): "
        "the element does not provide a callable to_dict() method.")
    assert not path.exists()


@pytest.mark.parametrize('container', ['line', 'env'])
@pytest.mark.parametrize('method', ['to_dict', 'to_json'])
def test_element_serialization_preserves_internal_error(container, method, tmp_path):
    error = AttributeError('Error inside element.to_dict')

    class BrokenElement(PythonElement):
        def to_dict(self):
            raise error

    env = xt.Environment()
    env.elements['broken'] = BrokenElement()
    line = env.new_line(components=['broken'])
    obj = line if container == 'line' else env
    with pytest.raises(AttributeError) as exc:
        if method == 'to_dict':
            obj.to_dict()
        else:
            obj.to_json(tmp_path / 'line.json')
    assert exc.value is error


def test_line_serialization_without_element_definitions():
    line = xt.Line(elements={'unsupported': PythonElement()})
    data = line.to_dict(include_element_dict=False)
    assert data['element_names'] == ['unsupported']
    assert 'elements' not in data
