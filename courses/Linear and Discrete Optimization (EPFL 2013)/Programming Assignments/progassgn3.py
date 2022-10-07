# improved implementation of the Simplex algorithm
from sympy import *

# requires a feasible basis B for the LP
def solve(A, b, c, B, verbose=False):
	m, n = A.rows, A.cols
	itr = 0

	AB = A.extract(B, range(n))
	bB = b.extract(B, [0])
	x = AB.LUsolve(bB)
	ABi = AB.inv()

	while True:
		itr += 1
		if verbose: print itr, x.T

		l = (c.T * ABi).T

		if all(e >= 0 for e in l):
			return x, itr

		# find leaving index B[r]
		r = min(i for i in range(l.rows) if l[i] < 0)

		d = -ABi[:, r]

		K = [i for i in range(m) if (A[i, :]*d)[0] > 0]

		if not K:
			return 'unbounded', itr

		# find entering index e
		e, v = None, None
		for k in K:
			w = (b[k] - (A[k, :] * x)[0]) / ((A[k, :] * d)[0])
			if v is None or w < v:
				v = w
				e = k

		# update basis
		B[r] = e
		AB[r, :] = A[e, :]
		bB[r, :] = b[e, :]

		# update inverse
		f = AB[r, :] * ABi
		g = -f
		g[r] = 1
		g /= f[r]
		
		X = eye(n)
		X[r, :] = g
		
		#ABi = ABi * X
		g[r] -= 1
		ABi += ABi[:, r] * g
		
		# move to the new vertex
		x = x + v * d

# small example
A = Matrix([[1, 0, 0, 0],
	[20, 1, 0, 0],
	[200, 20, 1, 0],
	[2000, 200, 20, 1],
	[-1, 0, 0, 0],
	[0, -1, 0, 0],
	[0, 0, -1, 0],
	[0, 0, 0, -1]])
b = Matrix([1,100,10000,1000000, 0,0,0,0])
c = Matrix([1000, 100, 10, 1])
B = range(4, 8)

#solve(A, b, c, B, verbose=True)


A = Matrix([[1,0,1],[1,1,0],[0,0,1]])
A = Matrix([[1,1,0,0,0],[1,0,1,0,1],[0,0,1,1,0],[0,0,0,0,1],[0,1,0,1,0]])

# domino
#def matching():
	
def compute_incidence_matrix(E, n, m):
	#A_G = zeros(n, m)
	A_G = zeros(n + m, m)
	e_i = 0
	for e in sorted(E):
		u, v = e
		A_G[u - 1, e_i] = A_G[v - 1, e_i] = 1
		e_i = e_i + 1
	A_G[n:, :] = -eye(m)
	return A_G
	
def find_edges(A, r):
	E = []
	n = len(A)
	for i in range(n):
		if i < n and A[i] != 0 and A[i + 1] != 0:
			E.append((A[i], A[i + 1]))
		if i + r < n and A[i] != 0 and A[i + r] != 0:
			E.append((A[i], A[i + r]))
	return E		
	
#	        1   2
#       3   4   5
#  6    7   8   9   10      11  12
# 13   14  15  16	17      18  19
# 20   21  22  23   24  25  26  27
#      28  29  30   31  32  33  34
#              35   36  27  38  39
#                   40  41  42
A = [
   0,  0,  1,  2,  0,  0,  0,  0,
   0,  3,  4,  5,  0,  0,  0,  0,
   6,  7,  8,  9, 10,  0, 11, 12,
  13, 14, 15, 16, 17,  0, 18, 19,
  20, 21, 22, 23, 24, 25, 26, 27,
  0,  28, 29, 30, 31, 32, 33, 34,
   0,  0,  0, 35, 36, 27, 38, 39,
   0,  0,  0,  0, 40, 41, 42,  0
   ]

E = find_edges(A, 8)
#print E
n = 42
m = len(E)
print '|V| = %d, |E| = %d' %(n, m)
A_G = compute_incidence_matrix(E, n, m)
#print A_G
b = ones(n + m, 1)
b[n:,:] = zeros(m, 1)
#print b
c = ones(m, 1)
B = range(m)
solve(A_G, b, c, B, verbose=True)