from collections import defaultdict

from ..line import Functions
import xdeps

class VarSharing:

    def __init__(
            self,
            lines,
            names,
            existing_manager=None,
            existing_vref=None,
            existing_eref=None,
            existing_fref=None,
    ):
        mgr = existing_manager or xdeps.Manager()

        if existing_vref is None:
            vref = mgr.ref(defaultdict(lambda: 0), "vars")
            vref._owner.default_factory = None
        else:
            vref = existing_vref

        if existing_fref is None:
            functions = Functions()
            fref = mgr.ref(functions, "f")
        else:
            fref = existing_fref

        if existing_eref is None:
            elements = {} # new root container
            eref = mgr.ref(elements, "eref") # new root ref
        else:
            eref = existing_eref

        self._vref = vref
        self._eref = eref
        self._fref = fref
        self.manager = mgr
        self.data = {}
        self.data['var_values'] = self._vref._owner

        if existing_manager is None:
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

            # Copy expressions
            self.manager.copy_expr_from(mgr1, "vars")
            self.manager.copy_expr_from(
                mgr1, "element_refs", {"element_refs": self._eref[name]})

        line._var_management = None

    def sync(self):
        for nn in self._vref._owner.keys():
            if (self._vref[nn]._expr is None
                and len(self._vref[nn]._find_dependant_targets()) > 1 # always contain itself
                ):

                self._vref[nn] = self._vref._owner[nn]

