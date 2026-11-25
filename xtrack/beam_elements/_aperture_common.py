# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

UNLIMITED = 1e10  # could use np.inf but better safe than sorry


class _HasInstalledPipeName:
    """Optional installed pipe placement name, stored with the Python element."""

    _noexpr_fields = {'installed_pipe_name'}
    _store_in_to_dict = ['installed_pipe_name']

    def __init__(self, *args, installed_pipe_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.installed_pipe_name = installed_pipe_name

    @property
    def installed_pipe_name(self):
        """Installed pipe placement names, comma-separated in beam order at transitions."""
        return getattr(self, '_installed_pipe_name', None)

    @installed_pipe_name.setter
    def installed_pipe_name(self, name):
        if name is not None and not isinstance(name, str):
            raise TypeError('installed_pipe_name must be a string or None')
        self._installed_pipe_name = name

    def __getstate__(self):
        return dict(element_state=super().__getstate__(), installed_pipe_name=self.installed_pipe_name)

    def __setstate__(self, state):
        # Older pickles contain only the xobject buffer and offset.
        if isinstance(state, dict) and 'element_state' in state:
            self.installed_pipe_name = state['installed_pipe_name']
            state = state['element_state']
        super().__setstate__(state)
