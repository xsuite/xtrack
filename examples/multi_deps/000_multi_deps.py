import json
import time
import xtrack as xt
import xpart as xp
import xdeps

from xdeps.tasks import ExprTask

def check_root_owner(t,ref):
    if hasattr(t,'_owner'):
        if t._owner is ref:
            return True
        else:
            return check_root_owner(t._owner,ref)
    else:
        return False

def iter_expr_tasks_owner(mgr,name):
  ref=mgr.containers[name]
  for t in mgr.find_tasks():
      #TODO check for all targets or limit to ExprTask
      if check_root_owner(t.taskid,ref):
          yield str(t.taskid),str(t.expr)


def load(self, dump, containers):
    """Reload the expressions in dump
    """
    for lhs, rhs in dump:
        lhs = eval(lhs, {}, containers)
        rhs = eval(rhs, {}, containers)
        task = ExprTask(lhs, rhs)
        self.register(task.taskid, task)


f1='../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json'
f4='../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b4.json'

line1 = xt.Line.from_dict(json.load(open(f1)))
line2 = xt.Line.from_dict(json.load(open(f4)))

mgr1=line1._var_management['manager']
mgr2=line2._var_management['manager']

# prepare manager with common vars and f
mgr=xdeps.Manager()
newv={}
newvref=mgr.ref(newv,'vars')
fref=mgr1.containers['f']
newfref=mgr.ref(fref._owner,'f')

#Load variables in common environment
vref=mgr1.containers['vars']
newv.update(vref._owner)
load(mgr,iter_expr_tasks_owner(mgr1,'vars'),{'vars':newvref,'f':newfref})
vref=mgr2.containers['vars']
newv.update(vref._owner)
load(mgr,iter_expr_tasks_owner(mgr2,'vars'),{'vars':newvref,'f':newfref})


#Prepare multi line
newe={}
neweref=mgr.ref(newe,'eref')

#Load elements in specific environment
eref=mgr1.containers['element_refs']
newe['lhcb1']=eref._owner
load(mgr,iter_expr_tasks_owner(mgr1,'element_refs'),{'vars':newvref,'f':newfref,'element_refs':neweref['lhcb1']})

#Load elements in specific environment
eref=mgr2.containers['element_refs']
newe['lhcb2']=eref._owner
load(mgr,iter_expr_tasks_owner(mgr2,'element_refs'),{'vars':newvref,'f':newfref,'element_refs':neweref['lhcb2']})

mgr.find_deps([mgr.containers['vars']['on_x1']])
mgr.containers['vars']['on_x1']=150

# test twiss
tracker1=line1.build_tracker()
tracker2=line2.build_tracker()

tw1=tracker1.twiss()
tw2=tracker2.twiss(reverse=True)

print(tw1['px'][np.array(tw1['name'])=='ip1'])
print(tw2['px'][np.array(tw2['name'])=='ip1'])


