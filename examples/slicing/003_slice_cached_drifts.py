import xtrack as xt

env = xt.Environment()
line = env.new_line(length=10)

# line.cut_at_s([2,5,7])

line.slice_thick_elements(
        slicing_strategies=[
            # Slicing with thin elements
            xt.Strategy(slicing=xt.Uniform(2), element_type=xt.Drift),])