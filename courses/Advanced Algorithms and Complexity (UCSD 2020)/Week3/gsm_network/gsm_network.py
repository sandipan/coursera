# python3
n, m = map(int, input().split())
edges = [ list(map(int, input().split())) for i in range(m) ]

# This solution prints a simple satisfiable formula
# and passes about half of the tests.
# Change this function to solve the problem.
def printEquisatisfiableSatFormula():
	print(3*m+n, 3*n)
	for u, v in edges:
		print(-u, -v, 0)
		print(-(n+u), -(n+v), 0)
		print(-(2*n+u), -(2*n+v), 0)
	for v in range(1, n+1):
		print(v, n+v, 2*n+v, 0)
    #print("3 2")
    #print("1 2 0")
    #print("-1 -2 0")
    #print("1 -2 0")

printEquisatisfiableSatFormula()
