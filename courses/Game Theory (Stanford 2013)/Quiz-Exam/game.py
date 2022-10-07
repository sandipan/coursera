#P1\P2	 1		2
#1		 a,b	 c,d
#2		 e,f	 g,h
def computeNashMixedStrategy(gmn, verbose=False):
	a, b = gmn[1, 1]
	c, d = gmn[1, 2]
	e, f = gmn[2, 1]
	g, h = gmn[2, 2]
	# p1 = (f - h) / (d + f - b - h)
	p1, p2 = abs(f - h - 0.0)/abs(d + f - b - h), abs(c - g - 0.0)/abs(c + e - a - g)
	if verbose:
		print 'p1=%.2f/%.2f=%.2f' %(abs(f - h), abs(d + f - b - h), p1)
		print 'p2=%.2f/%.2f=%.2f' %(abs(c - g), abs(c + e - a - g), p2)
	return (p1, p2)
	
gmn = {
(1, 1): (4, 2),
(1, 2): (5, 1),
(2, 1): (6, 0),
(2, 2): (3, 3)
}
gmn = {
(1, 1): (0, 1),
(1, 2): (1, 0),
(2, 1): (1, 0),
(2, 2): (0, 1)
}
gmn = {
(1, 1): (0, 1),
(1, 2): (1, 0),
(2, 1): (0.75, 0.25),
(2, 2): (0, 1)
}
#p1, p2 = computeNashMixedStrategy(gmn)
#print 'p1=%.2f, p2=%.2f' %(p1, p2)
	
# PS 2, Q1
gmn = {
(1, 1): (0.58, 0.42),
(1, 2): (0.95, 0.05),
(2, 1): (0.93, 0.07),
(2, 2): (0.70, 0.30)
}
# PS 2, Q2
#1\ 2	 Left	Right
#Left	 x,2	 0,0
#Right	 0,0	 2,2
gmn = {
(1, 1): (0, 2),
(1, 2): (0, 0),
(2, 1): (0, 0),
(2, 2): (2, 2)
}

# final
# Q5
gmn = {
(1, 1): (4, 4),
(1, 2): (1, 1),
(2, 1): (1, 1),
(2, 2): (2, 2)
}
p1, p2 = computeNashMixedStrategy(gmn)
print 'p1=%.2f, p2=%.2f' %(p1, p2)

#for i in range(32):
#	print 'p1=%.2f, p2=%.2f' %(computeNashMixedStrategy(gmn))
#	gmn[(1, 1)] = (i, 2)

import random	
# PS 2, Q3
#W1\ W2	 F1		 F2
#F1		 0,1	 1,1
#F2		 1,1	 1,0
gmn = {
(1, 1): (0, 1),
(1, 2): (1, 1),
(2, 1): (1, 1),
(2, 2): (1, 0)
}
n = 256
p1a = 0
p2a = 0
for trial in range(n):
	p1, p2 = computeNashMixedStrategy(gmn)
	#print 'p1=%.2f, p2=%.2f' %(p1, p2)
	i = random.randrange(2)
	gmn[(1, 1)] = (i, i^1)
	#i = random.randrange(2)
	gmn[(2, 2)] = (i^1, i)
	p1a += p1
	p2a += p2
#print p1a * 1.0 / n	
#print p2a * 1.0 / n	

# PS 2, Q4
#K\P	 XY		 Z
#XY		 2,9	 5,4
#Z		 5,4	 2,9
gmn = {
(1, 1): (2, 9),
(1, 2): (5, 4),
(2, 1): (5, 4),
(2, 2): (2, 9)
}
#computeNashMixedStrategy(gmn, verbose=True)

# PS 2, Q4
#K\P	 XY		 YZ		ZX
#X		 2,9	 5,4	2,9
#Y		 2,9	 2,9	5,4	
#Z		 5,4	 2,9	2,9

# Predator-Prey
gmn = {
(1, 1): (2, -5),
(1, 2): (3, -6),
(2, 1): (3, -2),
(2, 2): (-1, 0)
}
#computeNashMixedStrategy(gmn, verbose=True)

gmn = {
(1, 1): (2, 2),
(1, 2): (0, 2),
(2, 1): (1, 2),
(2, 2): (3, 3)
}
#computeNashMixedStrategy(gmn, verbose=True)

#udacity
gmn = {
(1, 1): (5, -5),
(1, 2): (3, -3),
(2, 1): (4, -4),
(2, 2): (2, -2)
}
#computeNashMixedStrategy(gmn, verbose=True)	# Dominant Strategy exists

gmn = {
(1, 1): (3, -3),
(1, 2): (6, -6),
(2, 1): (5, -5),
(2, 2): (4, -4)
}
#computeNashMixedStrategy(gmn, verbose=True)

# battel of the sexes 
gmn = {
(1, 1): (2, 1),
(1, 2): (0, 0),
(2, 1): (0, 0),
(2, 2): (1, 2)
}
#computeNashMixedStrategy(gmn, verbose=True)

def P1_dominates(s1, s2, p2s):
	foundDominant = True
	for s in p2s:
		if gmn[s1, s][0] <= gmn[s2, s][0]:
			foundDominant = False
			break
	return foundDominant	

def P2_dominates(s1, s2, p1s):
	foundDominant = True
	for s in p1s:
		if gmn[s, s1][1] <= gmn[s, s2][1]:
			foundDominant = False
			break
	return foundDominant	
	
def iterativeDominantStrategy(gmn, verbose=False):
	
	p1s =  set([ps[0] for ps in gmn.keys()])	
	p2s =  set([ps[1] for ps in gmn.keys()])	

	while True:
		foundDominant = False
		for s1 in p1s:
			if foundDominant:
				continue
			for s2 in p1s:
				if foundDominant or s1 == s2:
					continue
				foundDominant = P1_dominates(s1, s2, p2s)
				if foundDominant:
					print ('Removed strategy \'%c\' for Player1' %(s2))
					p1s = p1s.difference([s2])

		if not foundDominant:
			for s1 in p2s:
				if foundDominant:
					continue
				for s2 in p2s:
					if foundDominant or s1 == s2:
						continue
					foundDominant = P2_dominates(s1, s2, p1s)
					if foundDominant:
						print ('Removed strategy \'%c\' for Player2' %(s2))
						p2s = p2s.difference([s2])
		
		if not foundDominant:
			print 'No dominant pure strategy exists'
			break
			
		if len(p1s) == 1 and len(p2s) == 1:
			print list(p1s.union(p2s))
			break
			
gmn = {
('U', 'u'): (2,1), ('U', 'm'): (5,3), ('U', 'd'): (3,1),
('M', 'u'): (6,7), ('M', 'm'): (2,10), ('M', 'd'): (0,0),
('D', 'u'): (5,0), ('D', 'm'): (1,1), ('D', 'd'): (2,4)
}
#iterativeDominantStrategy(gmn)
gmn = {
('U', 'L'):(3,8), ('U', 'M'):(2,0), ('U', 'R'): (1,2),
('D', 'L'):(0,0), ('D', 'M'):(1,7), ('D', 'R'): (8,2)
}
#iterativeDominantStrategy(gmn)

# final
# Q4
gmn = {
('a', 'x'): (2,5), ('a', 'y'): (2,1), ('a', 'z'): (0,1),
('b', 'x'): (3,2), ('b', 'y'): (4,4), ('b', 'z'): (1,1),
('c', 'x'): (1,0), ('c', 'y'): (1,1), ('c', 'z'): (1,2)
}
iterativeDominantStrategy(gmn)

def maxmin(gmn, verbose=False):
	a, x = gmn[1, 1]
	b, x = gmn[2, 1]
	c, x = gmn[1, 2]
	d, x = gmn[2, 2]
	p = (1.0 * (d - b)) / (a - b - c + d)
	print ('p = %.2f / %.2f = %.2f, g = %.2f' %(abs(d - b), abs(a - b - c + d), p, a * p + b * (1 - p)))
	q = 0
	while q <= 1.0:
		print ('%.2f, %.2f, %.2f' %(q, a * q + b * (1 - q), c * q + d * (1 - q)))
		q += 0.05
		
gmn = {
(1, 1): (0.95, 0),
(1, 2): (1, 0),
(2, 1): (1, 0),
(2, 2): (0, 0)
}	
#maxmin(gmn)
# 1: M, 2: H
gmn = {
(1, 1): (3,0),
(1, 2): (1,2),
(2, 1): (2,1),
(2, 2): (0,3)
}	
#maxmin(gmn)
gmn = {
(1, 1): (2, -2),
(1, 2): (-2, 2),
(2, 1): (-2, 2),
(2, 2): (2, -2)
}	
#maxmin(gmn)

neg_inf = float('-inf')		

class Tree(object):
    def __init__(self, p=None, u=[neg_inf,neg_inf]):
		self.child = {}
		self.player = p
		self.utility = u
		
def backward_induction(h, p1, p2):
	if h.player == None: # terminal
		return h.utility
	best_util = [neg_inf, neg_inf]
	for a in h.child.keys():
		util_at_child = backward_induction(h.child[a], p1, p2)
		if util_at_child[h.player] > best_util[h.player]:
			best_util = util_at_child
			if h.player == 0:
				p1.append(a)
			else:
				p2.append(a)
	return best_util

root = Tree(p=0)
root.child['S'] = Tree(u=[0,2])
root.child['E'] = Tree(p=0)
root.child['E'].child['F'] = Tree(p=1)
root.child['E'].child['A'] = Tree(p=1)
root.child['E'].child['F'].child['F'] = Tree(u=[-2,-1])
root.child['E'].child['F'].child['A'] = Tree(u=[1,-2])
root.child['E'].child['A'].child['F'] = Tree(u=[-2,-1])
root.child['E'].child['A'].child['A'] = Tree(u=[3,1])
p1 = []
p2 = []
#print backward_induction(root, p1, p2)
#print p1
#print p2

root = Tree(p=0)
root.child['A'] = Tree(p=1)
root.child['A'].child['C'] = Tree(u=[3,8])
root.child['A'].child['D'] = Tree(u=[8,3])
root.child['B'] = Tree(p=1)
root.child['B'].child['E'] = Tree(u=[5,5])
root.child['B'].child['F'] = Tree(p=1)
root.child['B'].child['F'].child['C'] = Tree(u=[2,10])
root.child['B'].child['F'].child['H'] = Tree(u=[1,0])
#print backward_induction(root)

root = Tree(p=0)
root.child['N'] = Tree(u=[1,1])
root.child['P'] = Tree(p=1)
root.child['P'].child['T'] = Tree(p=0)
root.child['P'].child['D'] = Tree(u=[0,2])
root.child['P'].child['T'].child['S'] = Tree(u=[10,0])
root.child['P'].child['T'].child['H'] = Tree(u=[5,5])
p1 = []
p2 = []
#print backward_induction(root, p1, p2)
#print (p1, p2)

# final
# Q6
root = Tree(p=0)
root.child['N'] = Tree(u=[0,5])
root.child['A'] = Tree(p=1)
root.child['A'].child['R'] = Tree(u=[5,0])
root.child['A'].child['F'] = Tree(u=[-2,-2])
p1 = []
p2 = []
print backward_induction(root, p1, p2)

# Q7
root = Tree(p=0)
root.child['N'] = Tree(u=[0,5])
root.child['A'] = Tree(p=1)
root.child['A'].child['F'] = Tree(u=[-2,-2])
p1 = []
p2 = []
print backward_induction(root, p1, p2)

root = Tree(p=1)
root.child['B'] = Tree(p=0)
root.child['N'] = Tree(p=0)
root.child['N'].child['N'] = Tree(u=[0,5])
root.child['N'].child['A'] = Tree(p=1)
root.child['N'].child['A'].child['R'] = Tree(u=[5,0])
root.child['N'].child['A'].child['F'] = Tree(u=[-2,-2])
root.child['B'].child['N'] = Tree(u=[0,5])
root.child['B'].child['A'] = Tree(p=1)
root.child['B'].child['A'].child['F'] = Tree(u=[-2,-2])
p1 = []
p2 = []
print backward_induction(root, p1, p2)

# P1 know P2 does not
def computeBayesianNashEquillibrium(g, p, m1, m2):
	gg = {}
	print ''
	for i in [(1,1),(1,2),(2,1),(2,2)]:
		row = '\t'
		for j in [1,2]:
			u1, u2 = g[1][i[0],j], g[2][i[1],j]
			gg[i,j] = p * u1[0] + (1 - p) * u2[0], p * u1[1] + (1 - p) * u2[1]
			row += str(gg[i, j]) + ' '
		print row
	print ''
	nash = list()
	for i in gg.keys():
		k1, k2 = i
		found = True
		for k in [(1,1),(1,2),(2,1),(2,2)]:
			if gg[k, k2][0] > gg[i][0]:
				found = False
				break
		for k in [1,2]:
			if gg[k1, k][1] > gg[i][1]:
				found = False
				break
		if found:
			(k1, k2), k3 = i
			nash.append(m1[k1] + m1[k2] + ',' + m2[k3])
	print nash
	
g = [
{}, # dummy
{
(1, 1): (1, 1),
(1, 2): (0, 0),
(2, 1): (0, 0),
(2, 2): (0, 0)
},	
{
(1, 1): (0, 0),
(1, 2): (0, 0),
(2, 1): (0, 0),
(2, 2): (2, 2)
}
]	
#computeBayesianNashEquillibrium(g, 0.5, {1:'U', 2:'D'}, {1:'L', 2:'R'})
g = [
{}, # dummy
{
(1, 1): (1, 1),
(1, 2): (0, 2),
(2, 1): (0, 2),
(2, 2): (1, 1)
},	
{
(1, 1): (2, 2),
(1, 2): (0, 1),
(2, 1): (4, 4),
(2, 2): (2, 3)
}
]	
#computeBayesianNashEquillibrium(g, 0.5, {1:'U', 2:'D'}, {1:'L', 2:'R'})
# s<w<M
s=1
w=2
M=5
g = [
{}, # dummy
{
(1, 1): (-w, -w),
(1, 2): (M, 0),
(2, 1): (0, M),
(2, 2): (0, 0)
},	
{
(1, 1): (M-s, -w),
(1, 2): (M, 0),
(2, 1): (0, M),
(2, 2): (0, 0)
}
]	
#computeBayesianNashEquillibrium(g, 0.5, {1:'A', 2:'N'}, {1:'A', 2:'N'})
w=2
g = [
{}, # dummy
{
(1, 1): (2, 0),
(1, 2): (2, 0),
(2, 1): (w, 2-w),
(2, 2): (2, 0)
},	
{
(1, 1): (1, 0),
(1, 2): (1, 0),
(2, 1): (w, 1-w),
(2, 2): (1, 0)
}
]	
#computeBayesianNashEquillibrium(g, 0.5, {1:'S', 2:'W'}, {1:'H', 2:'N'})
w=1
g = [
{}, # dummy
{
(1, 1): (2, 0),
(1, 2): (2, 0),
(2, 1): (w, 2-w),
(2, 2): (2, 0)
},	
{
(1, 1): (1, 0),
(1, 2): (1, 0),
(2, 1): (w, 1-w),
(2, 2): (1, 0)
}
]	
#computeBayesianNashEquillibrium(g, 0.5, {1:'S', 2:'W'}, {1:'H', 2:'N'})
# P2 know P1 does not
def computeBayesianNashEquillibrium(g, p, m1, m2):
	gg = {}
	print ''
	for i in [1,2]:
		row = '\t'
		for j in [(1,1),(1,2),(2,1),(2,2)]:
			u1, u2 = g[1][i,j[0]], g[2][i,j[1]]
			gg[i,j] = p * u1[0] + (1 - p) * u2[0], p * u1[1] + (1 - p) * u2[1]
			row += str(gg[i, j]) + ' '
		print row
	print ''
	nash = list()
	for i in gg.keys():
		k1, k2 = i
		found = True
		for k in [1,2]:
			if gg[k, k2][0] > gg[i][0]:
				found = False
				break
		for k in [(1,1),(1,2),(2,1),(2,2)]:
			if gg[k1, k][1] > gg[i][1]:
				found = False
				break
		if found:
			k1, (k2, k3) = i
			nash.append(m1[k1] + ',' + m1[k2] + m2[k3])
	print nash

g = [
{}, # dummy
{
(1, 1): (2, 1),
(1, 2): (0, 0),
(2, 1): (0, 0),
(2, 2): (1, 2)
},	
{
(1, 1): (2, 0),
(1, 2): (0, 2),
(2, 1): (0, 1),
(2, 2): (1, 0)
}
]	
#computeBayesianNashEquillibrium(g, 0.5, {1:'L', 2:'P'}, {1:'L', 2:'P'})
#computeBayesianNashEquillibrium(g, 0.25, {1:'L', 2:'P'}, {1:'L', 2:'P'})

# P2 know P1 does not
def computeBayesianNashEquillibrium(g, p, m1, m2):
	gg = {}
	print ''
	for i in [1,2,3]:
		row = '\t'
		for j in [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)]:
			u1, u2 = g[1][i,j[0]], g[2][i,j[1]]
			gg[i,j] = p * u1[0] + (1 - p) * u2[0], p * u1[1] + (1 - p) * u2[1]
			row += str(gg[i, j]) + ' '
		print row
	print ''
	nash = list()
	for i in gg.keys():
		k1, k2 = i
		found = True
		for k in [1,2,3]:
			if gg[k, k2][0] > gg[i][0]:
				found = False
				break
		for k in [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)]:
			if gg[k1, k][1] > gg[i][1]:
				found = False
				break
		if found:
			k1, (k2, k3) = i
			nash.append(m1[k1] + ',' + m2[k2] + m2[k3])
	print nash

g = [
{},
{(1,1):(0,0),(1,2):(-1,1),(1,3):(1,-1),
 (2,1):(1,-1),(2,2):(0,0),(2,3):(-1,1),
 (3,1):(-1,1),(3,2):(1,-1),(3,3):(0,0)},
{(1,1):(neg_inf,neg_inf),(1,2):(-1,1),(1,3):(neg_inf,neg_inf),
 (2,1):(neg_inf,neg_inf),(2,2):(0,0),(2,3):(neg_inf,neg_inf),
 (3,1):(neg_inf,neg_inf),(3,2):(1,-1),(3,3):(neg_inf,neg_inf)}
]

#computeBayesianNashEquillibrium(g, 1.0/3, {1:'R', 2:'P', 3:'S'}, {1:'R', 2:'P', 3:'S'})
#computeBayesianNashEquillibrium(g, 2.0/3, {1:'R', 2:'P', 3:'S'}, {1:'R', 2:'P', 3:'S'})

#final
# P2 know P1 does not
def computeBayesianNashEquillibriumF(g, p, m1, m2):
	gg = {}
	print ''
	for i in [1,2]:
		row = '\t'
		for j in [(1,1),(1,2),(2,1),(2,2)]:
			u1, u2 = g[1][i,j[0]], g[2][i,j[1]]
			gg[i,j] = p * u1[0] + (1 - p) * u2[0], p * u1[1] + (1 - p) * u2[1]
			row += str(gg[i, j]) + ' '
		print row
	print ''
	nash = list()
	for i in gg.keys():
		k1, k2 = i
		found = True
		for k in [1,2]:
			if gg[k, k2][0] > gg[i][0]:
				found = False
				break
		for k in [(1,1),(1,2),(2,1),(2,2)]:
			if gg[k1, k][1] > gg[i][1]:
				found = False
				break
		if found:
			k1, (k2, k3) = i
			nash.append(m1[k1] + ',' + m2[k2] + m2[k3])
	print nash

# Q9
g = [
{}, # dummy
{
(1, 1): (3, 1),
(1, 2): (0, 0),
(2, 1): (2, 1),
(2, 2): (1, 0)
},	
{
(1, 1): (3, 0),
(1, 2): (0, 1),
(2, 1): (2, 0),
(2, 2): (1, 1)
}
]	
computeBayesianNashEquillibriumF(g, 1.0/4, {1:'L', 2:'R'}, {1:'L', 2:'R'})
# Q10
g = [
{}, # dummy
{
(1, 1): (-1, 2),
(1, 2): (1, -2),
(2, 1): (0, 3),
(2, 2): (0, 3)
},	
{
(1, 1): (-1, 0),
(1, 2): (1, 2),
(2, 1): (0, 3),
(2, 2): (0, 3)
}
]	
computeBayesianNashEquillibriumF(g, 1.0/2, {1:'E', 2:'O'}, {1:'F', 2:'N'})
computeBayesianNashEquillibriumF(g, 1.0/4, {1:'E', 2:'O'}, {1:'F', 2:'N'})
computeBayesianNashEquillibriumF(g, 3.0/4, {1:'E', 2:'O'}, {1:'F', 2:'N'})
	
#udacity