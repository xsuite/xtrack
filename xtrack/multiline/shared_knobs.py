from ..line import mathfunctions
import xdeps

class VarSharing:

    def __init__(self, lines, names):


        mgr = xdeps.Manager()
        newvref = mgr.ref({}, "vars")
        newfref = mgr.ref(mathfunctions, "f")

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

        if (line._var_management is not None
            and line._var_management["manager"] is not None):
            mgr1 = line._var_management["manager"]

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

        if line._var_management is None:
            line._var_management = {'data': {}}

        line._var_management["manager"] = None # remove old manager
        line._var_management["lref"] = self._eref[name]
        line._var_management["vref"] = self._vref
        line._var_management["fref"] = self._fref
        line._var_management["data"]["var_values"] = self._vref._owner

    def sync(self):
        for nn in self._vref._owner.keys():
            if (self._vref[nn]._expr is None
                and len(self._vref[nn]._find_dependant_targets()) > 1 # always contain itself
                ):

                self._vref[nn] = self._vref._owner[nn]

