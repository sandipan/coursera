# python3
from sys import stdin

EPS = 1e-5 #6
PRECISION = 20

def SelectPivotElement(a, used_rows, used_columns):
    # This algorithm selects the first free element.
    # You'll need to improve it to pass the problem.
    pivot_element = Position(0, 0)
    while used_rows[pivot_element.row]:
        pivot_element.row += 1
    while used_columns[pivot_element.column]:
        pivot_element.column += 1
    return pivot_element

def SwapLines(a, b, used_rows, pivot_element):
	found = False
	if not a[pivot_element.row][pivot_element.column]: 
		for row in range(len(used_rows)):
			if not used_rows[row] and a[row][pivot_element.column]:
				found = True
				break
	if found:
		a[row], a[pivot_element.row] = a[pivot_element.row], a[row]
		b[row], b[pivot_element.row] = b[pivot_element.row], b[row]

def ProcessPivotElement(a, b, pivot_element):
	# Write your code here
	pivot = a[pivot_element.row][pivot_element.column]
	if pivot:
		a[pivot_element.row] = list(map(lambda x: x / pivot, a[pivot_element.row]))
		b[pivot_element.row] = b[pivot_element.row] / pivot
		
		for i in range(len(a)):
			if i == pivot_element.row:
				continue
			if a[i][pivot_element.column]:
				s = a[i][pivot_element.column]
				a[i] = [a[i][j] - a[pivot_element.row][j]*s for j in range(len(a[i]))]
				b[i] = b[i] - b[pivot_element.row]*s
			
def MarkPivotElementUsed(pivot_element, used_rows, used_columns):
    used_rows[pivot_element.row] = True
    used_columns[pivot_element.column] = True

def SolveEquation(equation):
    a = equation.a
    b = equation.b
    size = len(a)

    used_columns = [False] * size
    used_rows = [False] * size
    for step in range(size):
        pivot_element = SelectPivotElement(a, used_rows, used_columns)
        SwapLines(a, b, used_rows, pivot_element)
        ProcessPivotElement(a, b, pivot_element)
        MarkPivotElementUsed(pivot_element, used_rows, used_columns)
    #num_nz_rows_aug = sum([1 for i in range(size) if any(a[i] + [b[i]])]) 
    return b

def PrintColumn(column):
    size = len(column)
    for row in range(size):
        print("%.20lf" % column[row])

class Equation:
    def __init__(self, a, b):
        self.a = a
        self.b = b

class Position:
    def __init__(self, column, row):
        self.column = column
        self.row = row

def solve_diet_problem(n, m, A, b, c):  
	# Write your code here
	A_ = [[] for _ in range(m+1)]
	b_ = [0]*(m+1)
	for i in range(m):
		A_[i] = [0]*m
		A_[i][i] = -1
	A_[m] = [1]*m
	b_[m] = 10**9
	A += A_
	b += b_
	max_val = -float('Inf')
	opt_sol = None
	from itertools import combinations
	for rows in list(combinations(range(n + m + 1), m)):
		a = [[] for _ in range(m)]
		b_ = [0]*m
		for i in range(m):
			a[i] += A[rows[i]]
			b_[i] = b[rows[i]]
		#print(a, b_)
		equation = Equation(a, b_)
		solution = SolveEquation(equation)
		feasible = all([solution[i] > 0 or abs(solution[i]) < EPS  for i in range(m)]) #and (sum(solution) <= 10**9)
		if feasible:
			for i in range(n):
				diff = sum(A[i][j]*solution[j] for j in range(m)) - b[i]
				if abs(diff) <= EPS:
					diff = 0
				if diff > 0:
					feasible = False
					break
		if feasible:
			#print(solution)
			val = sum(c[j] * solution[j] for j in range(m))
			if val > max_val:
				max_val = val
				opt_sol = solution
	#print(sum(opt_sol), abs(sum(opt_sol) - 10**9) < EPS)   
	#unbounded = sum(opt_sol) >= 10**9 or abs(sum(opt_sol) - 10**9) < EPS # check if the infinity constraint is tight
	return [-1, None] if opt_sol == None else [1, None ] if (sum(opt_sol) >= 10**9 or abs(sum(opt_sol) - 10**9) < EPS) else [0, opt_sol]   # check if the infinity constraint is tight

def run_tests():
	from glob import glob
	files = glob('tests/*')
	files_answer = sorted(list(filter(lambda x: x.endswith('.a'), files)))
	files_input = sorted(list(set(files) - set(files_answer)))
	#print(len(files_input), len(files_answer))
	for i in range(len(files_input)):
		lines = open(files_input[i]).read().splitlines()
		n, m = map(int, lines[0].split())
		A = []
		for j in range(1,n+1):
		  A += [list(map(int, lines[j].split()))]
		b = list(map(int, lines[n+1].split()))
		c = list(map(int, lines[n+2].split()))

		anst, ansx = solve_diet_problem(n, m, A, b, c)
		
		result = ''
		if anst == -1:
		  result += "No solution\n"
		if anst == 0:  
		  result += "Bounded solution\n"
		  result += ' '.join(list(map(lambda x : '%.18f' % x, ansx))) + '\n'
		if anst == 1:
		  result += "Infinity\n"
		res = open(files_answer[i]).read()
		passed = (result == res)
		try:
			print(i, n, m, result, res, 'passed' if passed else 'failed') #matching, res, 
			if not passed:
				print(result)
				print(res)
				#break
		except Exception as e:
			print(str(e))

n, m = list(map(int, stdin.readline().split()))
A = []
for i in range(n):
  A += [list(map(int, stdin.readline().split()))]
b = list(map(int, stdin.readline().split()))
c = list(map(int, stdin.readline().split()))

anst, ansx = solve_diet_problem(n, m, A, b, c)

if anst == -1:
  print("No solution")
if anst == 0:  
  print("Bounded solution")
  print(' '.join(list(map(lambda x : '%.18f' % x, ansx))))
if anst == 1:
  print("Infinity")

	
#run_tests()
    
 
