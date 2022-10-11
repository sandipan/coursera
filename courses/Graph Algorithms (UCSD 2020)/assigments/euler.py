import sys
from graphviz import Digraph
id = 0
from random import random, randint, choice

def draw_graph(adj, colors, tour):
	global id
	s = Digraph('graph', format='jpg')
	#s.attr(compound='true')
	edges = set([])
	for u in range(len(adj)):
		s.node(str(u), style='filled', color=colors[u])
	for i in range(len(tour)-1):
		colors[tour[i], tour[i+1]] = colors[tour[i+1], tour[i]] = 'red'
	for u in range(len(adj)):
		for v in adj[u]:
			if not (u, v) in edges and not (v, u) in edges: 
				edges.add((u, v))
				s.edge(str(u), str(v), color=colors.get((u,v), 'lightgray'))
	s.attr(label=r'\n\nCompute Euler Tour (Circuit) in a Directed Graph with DFS (Recursive)\n found: {}, Tour = [{}]'.format(len(tour) > 0, ','.join(map(str, tour))))
	s.attr(fontsize='15')
	s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1

def count_in_out_degrees(adj):
	n = len(adj)
	in_deg, out_deg = [0]*n, [0]*n
	for u in range(n):
		for v in adj[u]:
			out_deg[u] += 1
			in_deg[v] += 1			
	return in_deg, out_deg
	
def get_start_if_Euler_tour_present(in_deg, out_deg):
	start, end, tour = None, None, True
	for i in range(len(in_deg)):
		d = out_deg[i] - in_deg[i]
		if abs(d) > 1:
			print(i, d)
			tour = False
			break
		elif d == 1:
			start = i
		elif d == -1:
			end = i
	tour = (start != None and end != None) or (start == None and end == None)
	if tour and start == None: # a circuit 
		start = 0
	return (tour, start)
	
def dfs(adj, v, out_deg, tour):
	colors[v] = 'gray'
	draw_graph(adj, colors, tour)
	while out_deg[v] > 0:
		print(v)
		out_deg[v] -= 1
		dfs(adj, adj[v][out_deg[v]], out_deg, tour)
	tour[:] = [v] + tour
	colors[v] = 'dimgray'
	draw_graph(adj, colors, tour)
	#print(tour)

def compute_Euler_tour(adj):
	n, m = len(adj), sum([len(adj[i]) for i in range(len(adj))])
	in_deg, out_deg = count_in_out_degrees(adj)
	tour_present, start = get_start_if_Euler_tour_present(in_deg, out_deg)
	if not tour_present:
		return None
	tour = []
	dfs(adj, start, out_deg, tour)
	#print(tour, len(tour))
	if len(tour) == m+1:
		return tour
	return None
	
def DegreesNbrs(edges, dir = True):
	vertices, indegrees, outdegrees, nbrs = set([]), {}, {}, {}
	for e in edges:
		e = e.replace('->', ' ')
		#print e
		u, vs = str.split(e)
		u = int(u)
		vertices.add(u)
		vs = vs.split(',')
		for v in vs:
			v = int(v) 
			vertices.add(v)
			outdegrees[u] = outdegrees.get(u, 0) + 1
			indegrees[v] = indegrees.get(v, 0) + 1
			nbrs[u] = nbrs.get(u, []) + [v]
			if not dir:
				nbrs[v] = nbrs.get(v, []) + [u]
	vertices = list(vertices)		
	return vertices, indegrees, outdegrees, nbrs

def read_file(filename):
	return [line.strip() for line in open(filename)]

lines = read_file('e2.txt') # e1.txt
vertices, indegrees, outdegrees, nbrs = DegreesNbrs(lines, dir = True)
print(set(range(max(vertices)+1)) - set(sorted(vertices)))
adj = [[] for _ in range(len(vertices))]
for i in range(len(vertices)):
	adj[i] += nbrs.get(i, [])

colors = {v:'floralwhite' for v in range(len(adj))}
print(compute_Euler_tour(adj))
