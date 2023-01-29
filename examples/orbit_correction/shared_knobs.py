import xdeps


class VarSharing:

    def __init__(self, lines, names):

        line0 = lines[0]
        mgr0 = line0._var_management["manager"]

        mgr = xdeps.Manager()
        newvref = mgr.ref({}, "vars")
        mgr.ref(mgr0.containers["f"]._owner, "f")

        newe = {} # new root container
        neweref = mgr.ref(newe, "eref") # new root ref

        self._vref = newvref
        self._eref = neweref
        self.manager = mgr

        for ll, nn in zip(lines, names):
            self.add_line(ll, nn)

    def add_line(self, line, name):

        mgr1 = line._var_management["manager"]

        self._vref._owner.update(mgr1.containers["vars"]._owner) # copy data
        self.manager.copy_expr_from(mgr1, "vars") # copy expressions

        # bind data with line.element_dict
        self._eref._owner[name] = mgr1.containers["element_refs"]._owner

        # copy expressions
        self.manager.copy_expr_from(mgr1, "element_refs",
                                {"element_refs": self._eref[name]})
