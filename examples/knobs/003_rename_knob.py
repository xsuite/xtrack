import xtrack as xt

env = xt.load('../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
line = env.lhcb1


def rename(env, old, new, verbose=False):
    mgr = env.ref_manager
    old_expr = env.ref[old]._expr
    old_expr_or_value = old_expr if old_expr is not None else env.ref[old]._value
    env.vars[new] = old_expr_or_value
    r_old = env.ref[old]
    r_new = env.ref[new]
    t_old = mgr.tasks.get(r_old)
    if t_old is not None:
        if verbose:
            print(f"replacing target in {t_old}")
        mgr.set_value(r_new, t_old.expr)
    for rt in list(env.ref_manager.rdeps[r_old]):
        tt = mgr.tasks[rt]
        old_expr = str(tt.expr)
        new_expr = old_expr.replace(str(r_old), str(r_new))
        if verbose:
            print(f"replacing {old_expr} with {new_expr}")
        mgr.set_value(rt, eval(new_expr, mgr.containers))

    env.vars.remove(old)


rename(env, "on_x1vs", "on_x1_v_pippo")
