import io
import json
import pandas as pd
import numpy as np

from .shared_knobs import VarSharing
import xobjects as xo
import xtrack as xt
import xfields as xf

class Multiline:

    def __init__(self, lines: dict, link_vars=True):
        self.lines = {}
        self.lines.update(lines)

        line_names = list(self.lines.keys())
        line_list = [self.lines[nn] for nn in line_names]
        if link_vars:
            self._var_sharing = VarSharing(lines=line_list, names=line_names)
        else:
            self._var_sharing = None

        for ll in line_list:
            ll._in_multiline = True

    def to_dict(self, include_var_management=True):

        '''
        Save the multiline to a dictionary.

        Parameters
        ----------
        include_var_management: bool
            If True, the variable management data is included in the dictionary.

        Returns
        -------
        dct: dict
            The dictionary with the multiline data.
        '''

        dct = {}
        if include_var_management:
            dct['_var_manager'] = self._var_sharing.manager.dump()
            dct['_var_management_data'] = self._var_sharing.data
        dct['lines'] = {}
        for nn, ll in self.lines.items():
            dct['lines'][nn] = ll.to_dict(include_var_management=False)

        if hasattr(self, '_bb_config') and self._bb_config is not None:
            dct['_bb_config'] = {}
            for nn, vv in self._bb_config.items():
                if nn == 'dataframes':
                    dct['_bb_config'][nn] = {}
                    for kk, vv in vv.items():
                        dct['_bb_config'][nn][kk] = vv.to_dict()
                else:
                    dct['_bb_config'][nn] = vv
        return dct

    @classmethod
    def from_dict(cls, dct):

        '''
        Load a multiline from a dictionary.

        Parameters
        ----------
        dct: dict
            The dictionary with the multiline data.

        Returns
        -------
        new_multiline: Multiline
            The multiline object.
        '''

        lines = {}
        for nn, ll in dct['lines'].items():
            lines[nn] = xt.Line.from_dict(ll)

        new_multiline = cls(lines=lines, link_vars=('_var_manager' in dct))

        if '_var_manager' in dct:
            for kk in dct['_var_management_data'].keys():
                new_multiline._var_sharing.data[kk].update(
                                                dct['_var_management_data'][kk])
            new_multiline._var_sharing.manager.load(dct['_var_manager'])

        if '_bb_config' in dct:
            new_multiline._bb_config = dct['_bb_config']
            for nn, vv in dct['_bb_config']['dataframes'].items():
                new_multiline._bb_config[
                    'dataframes'][nn] = pd.DataFrame(vv)

        return new_multiline

    def to_json(self, file, **kwargs):
        '''Save the multiline to a json file.

        Parameters
        ----------
        file: str or file-like object
            The file to save to. If a string is provided, a file is opened and
            closed. If a file-like object is provided, it is used directly.
        **kwargs: dict
            Additional keyword arguments are passed to the `Line.to_dict` method.
        '''

        if isinstance(file, io.IOBase):
            json.dump(self.to_dict(**kwargs), file, cls=xo.JEncoder)
        else:
            with open(file, 'w') as fid:
                json.dump(self.to_dict(**kwargs), fid, cls=xo.JEncoder)

    @classmethod
    def from_json(cls, file, **kwargs):
        '''Load a multiline from a json file.

        Parameters
        ----------
        file: str or file-like object
            The file to load from. If a string is provided, a file is opened and
            closed. If a file-like object is provided, it is used directly.
        **kwargs: dict

        Returns
        -------
        new_multiline: Multiline
            The multiline object.
        '''

        if isinstance(file, io.IOBase):
            dct = json.load(file)
        else:
            with open(file, 'r') as fid:
                dct = json.load(fid)

        return cls.from_dict(dct, **kwargs)

    def build_trackers(self, _context=None, _buffer=None, **kwargs):
        '''
        Build the trackers for the lines.

        Parameters
        ----------
        _context: xobjects.Context
            The context in which the trackers are built.
        _buffer: xobjects.Buffer
            The buffer in which the trackers are built.
        **kwargs: dict
            Additional keyword arguments are passed to the `Line.build_tracker`
            method.

        '''

        for nn, ll in self.lines.items():
            ll.build_tracker(_context=_context, _buffer=_buffer, **kwargs)

    def __getitem__(self, key):
        return self.lines[key]

    def __dir__(self):
        return list(self.lines.keys()) + object.__dir__(self)

    def __getattr__(self, key):
        if key in self.lines:
            return self.lines[key]
        else:
            raise AttributeError(f"Multiline object has no attribute `{key}`.")

    @property
    def vars(self):
        if self._var_sharing is not None:
            return self._var_sharing._vref

    def install_beambeam_interactions(self, clockwise_line, anticlockwise_line,
                                      ip_names,
                                      num_long_range_encounters_per_side,
                                      num_slices_head_on,
                                      harmonic_number, bunch_spacing_buckets,
                                      sigmaz):

        '''
        Install beam-beam elements in the lines. Elements are inserted in the
        lines in the appropriate positions. They are not configured and are kept
        inactive.

        Parameters
        ----------
        clockwise_line: xt.Line
            The line in which the beam-beam elements for the clockwise beam
            are installed.
        anticlockwise_line: xt.Line
            The line in which the beam-beam elements for the anticlockwise beam
            are installed.
        ip_names: list
            The names of the IPs in the lines around which the beam-beam
            elements need to be installed.
        num_long_range_encounters_per_side: dict
            The number of long range encounters per side for each IP.
        num_slices_head_on: int
            The number of slices to be used for  the head-on beam-beam interaction.
        harmonic_number: int
            The harmonic number of the machine.
        bunch_spacing_buckets: float
            The bunch spacing in buckets.
        sigmaz: float
            The longitudinal size of the beam.

        '''

        if isinstance(num_long_range_encounters_per_side, dict):
            num_long_range_encounters_per_side = [
                num_long_range_encounters_per_side[nn] for nn in ip_names]

        # Trackers need to be invalidated to add elements
        for nn, ll in self.lines.items():
            ll.unfreeze()

        circumference = self.lines[clockwise_line].get_length()
        assert np.isclose(circumference,
                    self.lines[anticlockwise_line].get_length(),
                    atol=1e-4, rtol=0)

        bb_df_cw, bb_df_acw = xf.install_beambeam_elements_in_lines(
            line_b1=self.lines[clockwise_line],
            line_b4=self.lines[anticlockwise_line],
            ip_names=ip_names,
            num_long_range_encounters_per_side=num_long_range_encounters_per_side,
            num_slices_head_on=num_slices_head_on,
            circumference=circumference,
            harmonic_number=harmonic_number,
            bunch_spacing_buckets=bunch_spacing_buckets,
            sigmaz_m=sigmaz)

        self._bb_config = {
            'dataframes': {
                'clockwise': bb_df_cw,
                'anticlockwise': bb_df_acw
            },
            'ip_names': ip_names,
            'clockwise_line': clockwise_line,
            'anticlockwise_line': anticlockwise_line,
        }

    def configure_beambeam_interactions(self, num_particles,
                                    nemitt_x, nemitt_y, crab_strong_beam=True):

        '''
        Configure the beam-beam elements in the lines.

        Parameters
        ----------
        num_particles: float
            The number of particles per bunch.
        nemitt_x: float
            The normalized emittance in the horizontal plane.
        nemitt_y: float
            The normalized emittance in the vertical plane.
        crab_strong_beam: bool
            If True, crabbing of the strong beam is taken into account.

        '''

        xf.configure_beam_beam_elements(
            bb_df_cw=self._bb_config['dataframes']['clockwise'].copy(),
            bb_df_acw=self._bb_config['dataframes']['anticlockwise'].copy(),
            tracker_cw=self.lines[self._bb_config['clockwise_line']].tracker,
            tracker_acw=self.lines[self._bb_config['anticlockwise_line']].tracker,
            num_particles=num_particles,
            nemitt_x=nemitt_x, nemitt_y=nemitt_y,
            crab_strong_beam=crab_strong_beam,
            ip_names=self._bb_config['ip_names'])

        self.vars['beambeam_scale'] = 1.0

        for nn in ['clockwise', 'anticlockwise']:
            line = self.lines[self._bb_config[f'{nn}_line']]
            df = self._bb_config['dataframes'][nn]

            for bbnn in df.index:
                line.element_refs[bbnn].scale_strength = self.vars['beambeam_scale']




