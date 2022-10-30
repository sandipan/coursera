# python3
import numpy as np
from graphviz import Graph
from pysat.solvers import Glucose3
import matplotlib.pylab as plt

id = 0
n, m = map(int, input().split())
edges = [ list(map(int, input().split())) for i in range(m) ]

def get_colors(assignments):
	all_colors = np.array(['red', 'green', 'blue'])
	colors = {}
	for v in range(n):
		colors[v+1] = all_colors[[assignments[v]>0, assignments[n+v]>0, assignments[2*n+v]>0]][0]
	return colors

def get_hamiltonian_path(assignments):
	path = [None]*n
	for i in range(n):
		for j in range(n):
			if assignments[i*n+j] > 0: # True
				path[i] = j+1
	return path
	
def print_clauses(clauses):
	for c in clauses:
		vars = []
		for v in c:
			vars.append('{}x{}'.format('¬' if v < 0 else '', abs(v)))
		print('(' + ' OR '.join(vars) + ')')
		
def print_SAT_solution(assignments):
	sol = ''
	for x in assignments:
		sol += 'x{}={} '.format(abs(x),x>0)
	print(sol)

def get_vars(vars, assignments):
	if len(assignments) == 0:
		html = "'''<<table>"
		for v in vars:
			html += '<tr><td>'
			html += str('{}x{}'.format('¬' if v < 0 else '', abs(v)))
			html += '</td></tr>'
		html += "</table>>'''"
	else:
		html = "'''<<table>"
		for x in assignments:
			html += '<tr><td>'
			html += 'x{}={} '.format(abs(x),x>0)
			html += '</td></tr>'
		html += "</table>>'''"	
	return html
	
def get_clauses(clauses):
	html = "'''<<table>"
	for c in clauses:
		html += '<tr>'
		for v in c:
			html += '<td>'
			html += str('{}x{}'.format('¬' if v < 0 else '', abs(v)))
			html += '</td>'
		html += '</tr>'
	html += "</table>>'''"
	return html

def draw_graph(n, edges, clauses, assignments=[], colors={}):
	global id
	s = Graph('graph_color', format='jpg')
	#s.attr(nodesep='3', ranksep='3') #size='5000,5000') #layout="neato",  rankdir='LR', size='8,5')
	with s.subgraph(name='G') as g:
		for u in range(1,n+1):
			g.node(str(u), style='filled', color=('lightgray' if colors.get(u, None) == None else colors[u]))
		for u, v in edges:
			g.edge(str(u), str(v)) 	
		g.attr(fontsize='15')
	with s.subgraph(name='VC') as v:
		#v.attr(rank='same')
		v.node('vars', eval(get_vars(range(1,3*n+1), assignments))) #, style='filled', color='mistyrose')
		v.node('clauses', eval(get_clauses(clauses))) #, style='filled', color='mistyrose')
		v.attr(fontsize='15')	
	s.attr(label='Graph')
	s.attr(fontsize='15')
	s.attr(label=r'\n\nGraph coloring with SAT solver')
	s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1

import networkx as nx

def draw_graph2(edges, path=[]):
	global id
	g = nx.DiGraph()
	for u, v in edges:
		g.add_edge(str(u), str(v))
		g.add_edge(str(v), str(u))
	
	plt.figure(figsize=(15,10))
	plt.subplot(121)
	pos = nx.kamada_kawai_layout(g) #spring_layout(g) fruchterman_reingold_layout
	nx.draw_networkx_nodes(g, pos, node_size=400, node_color='#00b4d9')
	nx.draw_networkx_labels(g, pos)
	nx.draw_networkx_edges(g, pos, alpha=0.4, arrows=False)
	if path:
		nx.draw_networkx_edges(g, pos, edgelist=[(str(path[i]), str(path[i+1])) for i in range(len(path)-1)], edge_color='r', arrows=True, arrowsize=50) #, arrowstyle='fancy')
	plt.title('Computing the Hamiltonian Path with SAT-solver', size=20)
	plt.savefig('out/graph_{:03d}.png'.format(id))
	plt.close()
	id += 1

# This solution prints a simple satisfiable formula
# and passes about half of the tests.
# Change this function to solve the problem.
def solve_graph_3coloring_satsolver():
	n_clauses = 3*m+n
	n_vars = 3*n
	clauses = []
	for u, v in edges:
		clauses.append((-u, -v)) # corresponding to red color
		clauses.append((-(n+u), -(n+v))) # corresponding to green color
		clauses.append((-(2*n+u), -(2*n+v))) # corresponding to blue color
	for v in range(1, n+1):
		clauses.append((v, n+v, 2*n+v)) # at least one color
	print_clauses(clauses)
	draw_graph(n, edges, clauses)
	g = Glucose3()
	for c in clauses:
		g.add_clause(c)
	status = g.solve()
	assignments = g.get_model()
	print(status)
	print_SAT_solution(assignments)
	colors = get_colors(assignments)
	draw_graph(n, edges, clauses, assignments, colors)	
		
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
		
	n_clauses = 2*n + (2*n*n-n-m)*(n-1)
	n_vars = n*n
	clauses = []
	print(n_clauses, n_vars)
	
	# i = pos, j = node
	for j in range(n):
		clause = []
		for i in range(n):
			clause.append(index(i,j))
		clauses.append(clause)
	for i in range(n):
		clause = []
		for j in range(n):
			clause.append(index(i,j)) 
		clauses.append(clause)
	for j in range(n):
		for i in range(n):
			for k in range(i+1, n):
				clauses.append((-index(i,j), -index(k,j)))
	for i in range(n):
		for j in range(n):
			for k in range(j+1, n):
				clauses.append((-index(i,j), -index(i,k)))
	for k in range(n-1):
		for i in range(n):
			for j in range(n):
				if i == j: continue
				if not [i+1, j+1] in edges:
					clauses.append((-index(k,i), -index(k+1,j)))
	print_clauses(clauses)
	draw_graph2(edges)
	g = Glucose3()
	for c in clauses:
		g.add_clause(c)
	status = g.solve()
	assignments = g.get_model()
	print(status)
	print_SAT_solution(assignments)
	path = get_hamiltonian_path(assignments)
	print(path)
	draw_graph2(edges, path)	
	
printEquisatisfiableSatFormula()

