import xtrack as xt

def load(file=None, string=None, format=None, timeout=5.):

    if format is None and file is not None:
        if file.endswith('.json'):
            format = 'json'
        elif file.endswith('.seq') or file.endswith('.madx') or file.endswith('.mad'):
            format = 'madx'
        elif file.endswith('.py'):
            format = 'python'

    if file.startswith('http://') or file.startswith('https://'):
        assert string is None, 'Cannot specify both fname and string'
        string = xt.general.read_url(file, timeout=timeout)
        file = None

    if file is not None:
        assert string is None, 'Cannot specify both fname and string'

    if string is not None:
        assert file is None, 'Cannot specify both fname and string'
        assert format is not None, 'Must specify format when using string'

    assert format in ['json', 'madx', 'python'], f'Unknown format {format}'

    if format == 'json':
        ddd = xt.json.load(file=file, string=string)
        if '__class__' in ddd:
            cls_name = ddd.pop('__class__')
            cls = getattr(xt, cls_name)
            return cls.from_dict(ddd)
        elif 'lines' in ddd: # is environment
            return xt.Environment.from_dict(ddd)
        elif 'element_names' in ddd:
            return xt.Line.from_dict(ddd).env
        else:
            raise ValueError('Cannot determine class from json data')
    elif format == 'madx':
        return xt.load_madx_lattice(file=file, string=string)
    elif format == 'python':
        if string is not None:
            raise NotImplementedError('Loading from string not implemented for python format')
        env = xt.Environment()
        env.call(file)
        return env
    else:
        raise ValueError(f'Unknown format {format}')
