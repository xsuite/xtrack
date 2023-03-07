import pandas as pd
import numpy as np

from .shared_knobs import VarSharing
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

    def build_trackers(self, **kwargs):
        for nn, ll in self.lines.items():
            ll.build_tracker(**kwargs)

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




