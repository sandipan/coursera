# python3
n, m = map(int, input().split())
edges = [ list(map(int, input().split())) for i in range(m) ]

# This solution prints a simple satisfiable formula
# and passes about half of the tests.
# Change this function to solve the problem.
def printEquisatisfiableSatFormula():
	global edges
	redges = []
	for e in edges:
		if not e[::-1] in edges: 
			redges.append(e[::-1])
	edges += redges
	#print(len(edges))
	m = len(edges)
	
	def index(i, j):
		return n*i + j + 1
		
	print(2*n + (2*n*n-n-m)*(n-1), n*n)
	# i = pos, j = node
	count = 0
	for j in range(n):
		clause = ''
		for i in range(n):
			clause += str(index(i,j)) + ' '
		clause += '0'
		print(clause)
		count += 1
	for i in range(n):
		clause = ''
		for j in range(n):
			clause += str(index(i,j)) + ' '
		clause += '0'
		print(clause)
		count += 1
	#print(count)
	for j in range(n):
		clause = ''
		for i in range(n):
			for k in range(i+1, n):
				clause = str(-index(i,j)) + ' ' + str(-index(k,j)) + ' '
				clause += '0'
				print(clause)
				count += 1
	#print(count)
	for i in range(n):
		clause = ''
		for j in range(n):
			for k in range(j+1, n):
				clause = str(-index(i,j)) + ' ' + str(-index(i,k)) + ' '
				clause += '0'
				print(clause)
				count += 1
	#print(count)
	for k in range(n-1):
		clause = ''
		for i in range(n):
			for j in range(n):
				if i == j: continue
				if not [i+1, j+1] in edges:
					clause = str(-index(k,i)) + ' ' + str(-index(k+1,j)) + ' '
					clause += '0'
					print(clause)
					count += 1
		#print()
	#print(count)
	#for u, v in edges:
	#	print(-u, -v, 0)
    #print("3 2")
    #print("1 2 0")
    #print("-1 -2 0")
    #print("1 -2 0")

printEquisatisfiableSatFormula()
