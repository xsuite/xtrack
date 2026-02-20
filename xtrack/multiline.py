from warnings import warn

from .environment import Environment

# For backward compatibility
class Multiline(Environment):

    def __init__(self, *args, **kwargs):
        warn('Multiline is deprecated, use Environment instead', FutureWarning)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_dict(cls, dct, **kwargs):
        if 'xsuite_data_type' in dct and dct['xsuite_data_type'] == 'Environment':
            warn(
                'Loading an environment through a deprecated Multiline: consider loading the dictionary directly.',
                FutureWarning
            )
            # TODO: Needs to be sorted, returns environment
            return super().from_dict(dct, **kwargs)
        else:
            raise ValueError('Loading legacy Multiline from dict is not supported anymore. ')
