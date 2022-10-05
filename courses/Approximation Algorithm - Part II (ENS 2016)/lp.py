'''
from pulp import * 

prob = LpProblem("test1", LpMinimize) 

# Variables 
x = LpVariable("x", 0, 4) 
y = LpVariable("y", -1, 1) 
z = LpVariable("z", 0) 

# Objective 
prob += x + 4*y + 9*z 

# Constraints 
prob += x+y <= 5 
prob += x+z >= 10 
prob += -y+z == 7 

GLPK().solve(prob) 

# Solution 
for v in prob.variables(): 
	print v.name, "=", v.varValue 

print "objective=", value(prob.objective)  
'''

from pulp import * 

#Week1
prob = LpProblem("quiz2", LpMinimize) 

# Variables 
x1 = LpVariable("x1", 0) 
x2 = LpVariable("x2", 0) 
x3 = LpVariable("x3", 0) 

# Objective 
prob += 10*x1 + 5*x2 + 4*x3 

# Constraints 
prob += x1+x2+x3 >= 10 
prob += x1-x3 >= 2 
prob += -5*x1+x2-2*x3>=4
prob += 6*x1-x2+x3>=8

GLPK().solve(prob) 

# Solution 
for v in prob.variables(): 
	print v.name, "=", v.varValue 

print "objective=", value(prob.objective)  
