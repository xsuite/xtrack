import xtrack as xt

def load(fname=None, string=None, format=None, timeout=1.):

    if format is None and fname is not None:
        if fname.endswith('.json'):
            format = 'json'
        elif fname.endswith('.seq') or fname.endswith('.madx') or fname.endswith('.mad'):
            format = 'madx'
        elif fname.endswith('.py'):
            format = 'python'

    if fname.startswith('http://') or fname.startswith('https://'):
        assert string is None, 'Cannot specify both fname and string'
        string = xt.general.read_url(fname, timeout=timeout)
        fname = None

    if fname is not None:
        assert string is None, 'Cannot specify both fname and string'

    if string is not None:
        assert fname is None, 'Cannot specify both fname and string'
        assert format is not None, 'Must specify format when using string'

    assert format in ['json', 'madx', 'python'], f'Unknown format {format}'

    if format == 'json':
        ddd = xt.json.load(file=fname, string=string)
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
        return xt.load_madx_lattice(fname=fname, string=string)
    elif format == 'python':
        if string is not None:
            raise NotImplementedError('Loading from string not implemented for python format')
        env = xt.Environment()
        env.call(fname)
        return env
    else:
        raise ValueError(f'Unknown format {format}')
