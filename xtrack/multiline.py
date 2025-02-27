from .environment import Environment
from .multiline_legacy import MultilineLegacy
import xtrack as xt

# For backward compatibility
class Multiline(Environment):

    def __init__(self, *args, **kwargs):
        print('Warning: Multiline is deprecated, use Environment instead')
        super().__init__(*args, **kwargs)

    @classmethod
    def from_dict(cls, dct, **kwargs):
        if 'xsuite_data_type' in dct and dct['xsuite_data_type'] == 'Environment':
            # TODO: Needs to be sorted, returns environment
            return super().from_dict(dct, **kwargs)
        else:
            out = cls._from_legacy_multiline_dict(dct)
            print('\nWarning: you seem to be loading a legacy multiline file. '
                  'The `Multiline` class is deprecated and is now replaced by `Environment`. '
                  'Your multiline has been converted automatically to an Environment object. '
                  '\nThis file will become unreadable in the future. We recommend to save it '
                  'as an Environment object. This can be easily done as follows:\n\n'
                  '    import xtrack as xt\n'
                  '    env = xt.Multiline.from_json("my_old_multiline.json")\n'
                  '    env.to_json("my_new_environment.json")\n')
            return out

    @classmethod
    def _from_legacy_multiline_dict(cls, dct):

        lines = {}

        for line_name in dct['lines']:

            dct_line = dct['lines'][line_name].copy()

            new_man_data = []
            for ee in dct['_var_manager']:
                new_ee = []
                skip = False
                for cc in ee:
                    if 'eref' in cc and f"eref['{line_name}']" not in cc:
                        skip = True
                        break
                    new_cc = cc.replace(f"eref['{line_name}']", 'element_refs')
                    new_ee.append(new_cc)

                if skip:
                    continue

                new_man_data.append(tuple(new_ee))

            dct_line['_var_management_data'] = {}
            dct_line['_var_management_data']['var_values'] = dct['_var_management_data']['var_values'].copy()
            dct_line['_var_manager'] = new_man_data

            line = xt.Line.from_dict(dct_line)

            lines[line_name] = line

        env = xt.Environment(lines=lines)
        if 'metadata' in dct:
            env.metadata.update(dct['metadata'])

        return env
