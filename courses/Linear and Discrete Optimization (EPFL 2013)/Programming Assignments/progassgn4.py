from sympy import *
import math

# returns a vector from the kernel of a matrix with n columns
def kernel(A, n):
    if A.rows == 0:
        return Matrix([1]*n)

    R, pivots = A.rref()
    free = list(set(range(n)) - set(pivots))

    Ap = A.extract(range(A.rows), pivots)

    bp = Matrix([0]*Ap.rows)
    for i in free:
        bp -= A[:, i]

    xp = Ap.LUsolve(bp)
    x = [1]*n
    for i in range(len(pivots)):
        x[pivots[i]] = xp[i]
    return Matrix(x)

# takes a set of row indices I of the matrix A and
# returns a subset B of I such that
# i) rows of A_B are linearly independent
# ii) rows of A_B have the same span as those of A_I
def filterredundant(I, A):
    R, pivots = A.extract(I, range(A.cols)).T.rref()
    B = [I[i] for i in pivots]
    return B

# returns a vertex from P = {x | Ax <= b}
# assuming that x0 is feasible for P
def rayshooting(A, b, x0, verbose=False):
    m, n = A.rows, A.cols
    itr = 0

    # check rank of A
    R, pivots = A.rref()
    if len(pivots) < n:
        print 'Given polyhedron has no vertex'
        return None

    # add constraints that are active at x0
    C = [i for i in range(m) if (A[i, :]*x0)[0] == b[i]]

    # maintain a global set of active constraints
    G = set(C)

    # filter redundant active inequalities at B to ensure B is a basis
    B = filterredundant(C, A)

    AB = A.extract(B, range(n))
    
    # if x0 is not a vertex then perform ray shooting
    while AB.rows < n:
    
        if verbose: print 'Point', itr, ':', x0.T
        
		# choose a direction
        d = kernel(AB, n)

        # compute the indices of potential inequalities that may be
        # violated when moving in the direction d
        K = [i for i in range(m) if i not in G and (A[i, :]*d)[0] > 0]
        if not K: # if K is empty
            d = -d
            K = [i for i in range(m) if i not in G and (A[i, :]*d)[0] > 0]

        # compute epsilon
        L = [((b[k] - (A[k, :] * x0)[0])/((A[k, :] * d)[0]), k) for k in K]
        epsilon = min(L)[0]

        # move to a new point
        x0 = x0 + epsilon * d

        # find new active constraints at the new point
        C = [i for i in range(m) if i not in G and (A[i, :]*x0)[0] == b[i]]

        # add to global set of active constraints
        G.update(set(C))

        # add to basis
        B += C
        B = filterredundant(B, A)

        # update AB
        AB = A.extract(B, range(n))
        itr += 1

    # find the vertex corresponding to the feasible basis AB
    bB = b.extract(B, [0])
    x = AB.LUsolve(bB)

    print '\n', list(x), 'is a vertex of the polytope'
    print 'Found in', itr, 'iterations'
    return x

if __name__ == '__main__':

    # simple example
    A = Matrix([[-4, 1], [1, 2], [2, 1], [5, -2], [-4, -2]])
    b = Matrix([-4, 10, 11, 23, -4])
    c = Matrix([1, 1])
    x0 = Matrix([3, 1])

    rayshooting(A, b, x0, verbose=True)
