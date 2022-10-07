# solves *bounded* LPs of the form:
# max cx
# sub to: Ax <= b

from sympy import *
from itertools import combinations

# enumerates all the vertices of {x | Ax <= b}
def enumeratevertices(A, b):
    m, n = A.rows, A.cols

    for rowlist in combinations(range(m), n):
        Ap = A.extract(rowlist, range(n)) #
        bp = b.extract(rowlist, [0]) #

        if Ap.det() != 0: #
            xp = Ap.LUsolve(bp) #

            d = A * xp - b
            feasible = True
            for i in range(m): #
                if d[i] > 0: #
                    feasible = False

            if feasible:
                yield xp

# finds the optimum using vertex enumeration
def findoptimum(A, b, c):
    m, n = A.rows, A.cols

    bestvalue, bestvertex = None, None
    for vertex in enumeratevertices(A, b):
        print (vertex.T, (vertex.T * c)[0])
        if not bestvalue or (vertex.T * c)[0] > bestvalue: #
            bestvalue = (vertex.T * c)[0]
            bestvertex = vertex

    return bestvertex

def solve(A, b, c):
    x = findoptimum(A, b, c)

    if not x:
        print 'LP is infeasible'
    else:
        print 'Vertex', x.T, 'is optimal'
        #print 'Optimal value is', (x.T * c)[0] #
        print 'Optimal value is', (c.T * x)[0] #

if __name__ == '__main__':

    # small example
    A = Matrix([[50, 24], 
         [30, 33],
         [-1, 0],
         [0, -1]])
    b = Matrix([2400, 2100, -45, -5])
    c = Matrix([1, 1])

    solve(A, b, c)
	
    A = Matrix([[-10,  -6, -9, -10],
                [  8,  -6, -5,  -5],
                [ -7,  -1, -9,   3],
                [ -1,  -4,  5,  10],
                [  1,   2,  0,  10],
                [  2,  -9,  3,  -8],
                [ -8,  -1, -8,   1],
                [  7, -10,  4,  -4],
                [-10,   2,  5,   8],
                [ -7,   9,  4,  -4],
                [ -1,   0,  0,   0],
                [  0,  -1,  0,   0],
                [  0,   0, -1,   0],
                [  0,   0,  0,  -1]])
    b = Matrix([9, 7, 3, 4, 8, 0, 3, 2, 4, 8, 0, 0, 0, 0])
    c = Matrix([2, -2, -3, 8])

    solve(A, b, c)	