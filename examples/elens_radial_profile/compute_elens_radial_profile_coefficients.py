import numpy as np
import pandas as pd
import random
import decimal
import sys 

# assumes that the following format is given:
# r.mm            j.Acm2
# only two columns 
# argument for the script : 
# inner_radius elens
# outer_radius elens
# inner_radius measurement
# outer_radius measurement 

# for example 1.4, 2.8, 4, 8


#load the measured profile
data = pd.read_table('measured_radial_profile.dat')

del(data['Unnamed: 1'])

r1_lens, r2_lens = float(sys.argv[1]), float(sys.argv[2])
r1_meas, r2_meas = float(sys.argv[3]), float(sys.argv[4])



#radius in mm
def compute_coef(r_measured, j_measured, r_1_new, r_2_new, r_1_old, r_2_old, p_order = 13):
    new_r = r_measured*(r_2_new-r_1_new)/(r_2_old-r_1_old)   
    new_j = j_measured*(r_2_old-r_1_old)/(r_2_new-r_1_new) 
    
    product = new_r*new_j
    
    delta_r = new_r[2]-new_r[1]
    
    numerator = [];
    s = len(new_r)
    for i in range(s):
        numerator.append((delta_r*max(np.cumsum(product[0:i+1])))) 
    L = np.cumsum(product)
    denominator = max(L)*delta_r 
    f_r = np.array(numerator/denominator)
    r_selected = new_r[new_j != 0]
    f_selected = f_r[new_j != 0]
    coef = np.polyfit(r_selected, f_selected, p_order)
    #D = decimal.Decimal
    #for i in range(len(coef)):
        #print(D(coef[i]))
    return coef

#radius in mm
r = data['r.mm']
j = data['j.Acm2']
C = compute_coef(r, j, r1_lens, r2_lens, r1_meas, r2_meas)

polynomial_string="["
for c in C:
    polynomial_string = polynomial_string + "{0},".format(c) 

polynomial_string=polynomial_string[0:-1] + "]"

print("xt.Elens(current=5, inner_radius={0}, outer_radius={1},".format(r1_lens, r2_lens))
print("elens_length=3, voltage=10, coefficients_polynomial={0},polynomial_order={1})".format(polynomial_string, len(C)))

