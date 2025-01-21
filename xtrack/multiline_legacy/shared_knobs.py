from collections import defaultdict

from ..line import Functions
import xdeps

class VarSharing:

    def __init__(self, lines, names):


        mgr = xdeps.Manager()
        newvref = mgr.ref(defaultdict(lambda: 0), "vars")
        newvref._owner.default_factory = None

        functions = Functions()
        newfref = mgr.ref(functions, "f")

        newe = {} # new root container
        neweref = mgr.ref(newe, "eref") # new root ref

        self._vref = newvref
        self._eref = neweref
        self._fref = newfref
        self.manager = mgr
        self.data = {}
        self.data['var_values'] = self._vref._owner

        for ll, nn in zip(lines, names):
            self.add_line(ll, nn, update_existing=False)

        self.sync()

    def add_line(self, line, name, update_existing=False):

        # bind data with line.element_dict
        self._eref._owner[name] = line.element_dict

        if (hasattr(line, "_var_management")
                and line._var_management is not None
                and line._var_management["manager"] is not None):

            mgr1 = line._var_management["manager"]
            if len(mgr1.containers["f"]._owner._funcs.keys()) > 0:
                raise NotImplementedError("Functions not supported yet in multiline")

            if update_existing:
                self._vref._owner.update(mgr1.containers["vars"]._owner) # copy data
            else:
                # Find new variables
                new_vars =(set(mgr1.containers["vars"]._owner.keys())
                            - set(self._vref._owner.keys()))
                # Add new variables
                for vv in new_vars:
                    self._vref._owner[vv] = mgr1.containers["vars"]._owner[vv]

            self.manager.copy_expr_from(mgr1, "vars") # copy expressions

            # copy expressions
            self.manager.copy_expr_from(mgr1, "element_refs",
                                    {"element_refs": self._eref[name]})

        line._var_management = None

    def sync(self):
        for nn in self._vref._owner.keys():
            if (self._vref[nn]._expr is None
                and len(self._vref[nn]._find_dependant_targets()) > 1 # always contain itself
                ):

                self._vref[nn] = self._vref._owner[nn]

