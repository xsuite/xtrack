import numpy as np
import xtrack as xt

line = xt.Line.from_json(
    '../../test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json')

line.build_tracker()

class CacheItem:
    def __init__(self, name, index=None, line=None):
        self.name = name
        self.index = index

        assert line is not None

        all_names = line.element_names
        mask = np.zeros(len(all_names), dtype=bool)
        setter_names = []
        for ii, nn in enumerate(all_names):
            ee = line.element_dict[nn]
            if hasattr(ee, '_xobject') and hasattr(ee._xobject, name):
                mask[ii] = True
                setter_names.append(nn)

        multisetter = xt.MultiSetter(line=line, elements=setter_names, field=name)
        self.names = setter_names
        self.multisetter = multisetter
        self.mask = mask

    def get_full_array(self):
        full_array = np.zeros(len(self.mask), dtype=np.float64)
        full_array[self.mask] = self.multisetter.get_values()
        return full_array

hxl_cache = CacheItem(name='hxl', line=line)

class CacheForLine:

    def __init__(self, line, fields):
        self.line = line
        self.fields = fields
        self._cache = {}

        for ff in fields:
            if isinstance(ff, str):
                name = ff
                index = None
            else:
                name, index = ff
            self._cache[name] = CacheItem(name=name, index=index, line=line)

    def __getitem__(self, key):
        return self._cache[key].get_full_array()

cache = CacheForLine(line=line, fields=['hxl', 'hyl', 'length'])