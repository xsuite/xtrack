import xtrack as xt
import numpy as np
env = xt.Environment()

# Variables
env["a"] = 3
env["b"] = np.array([1, 3])        # to_dict issue, json reconstruc np.array natively
env["fun"]= math.sin

env.ref["b"][0] = env.ref["a"] * 3 #
env.set("b[0]", "fun(a*3)")        #
env["c"] = env.ref["b"][0] * 3     #

#Elements
env.new("mq0","Quadrupole",l="b[0]")
env.ref["mq0"]

#Lines
env["b1"]     # line
env.b1        # line has the right to pollute namespace
env.ref["b1"] # ref of a line

# General
env[] -> Python values
env.ref -> reference

# METADATA
env["mq"].metadata={"slot_id":123123,"layout_id":21332141}
# OR
env.extra_element_attributes=["slot_id","layout_id","polarity"]

#Containenr-like with benefits
env.vars   # references duplicated by ref
env.elems  # values like element_dict
env.lines  # lines values

env.vars.get_table()
env.funcs.get_table()
env.elems.get_table()
env.lines.get_table()
#introduce version and metadata on json

#--------------------------------------------------

#to be deprecated but supported
env.vv
env.element_refs


