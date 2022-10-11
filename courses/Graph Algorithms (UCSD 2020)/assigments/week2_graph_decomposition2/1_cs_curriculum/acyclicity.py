#Uses python3

import sys
from graphviz import Digraph
id = 0
from random import random, randint, choice
from copy import deepcopy

def gen_ER_graph(n, p):
	adj = [[] for _ in range(n)]
	for u in range(n):
		for v in range(u+1, n):
			if random() < p:
				if random() < 0.8:
					adj[u].append(v)
				else:
					adj[v].append(u)
	print(adj)
	return adj
	
def gen_cycle(n):
	adj = [[] for _ in range(n)]
	nodes = [0] #randint(0, n-1)]
	for i in range(1, n):
		#if not i in nodes:
		j = choice(nodes)
		adj[j].append(i)
		#adj[i].append(j)
		nodes.append(i)
	#adj1 = deepcopy(adj)
	i = choice(nodes)
	while not i or i in adj[0]:
		i = choice(nodes)
	#j = choice(nodes)
	#print(nodes, i, j)
	adj[i].append(0)
	#if i > j:
	#	i, j = j, i
	#adj[i].append(j)
	#adj1[j].append(i)
	print(adj)
	#print(adj1)
	return adj #, adj1
	
def get_stack(stack):
	html = "'''<<table>"
	n =  len(stack) if stack else 0
	for i in range(10-n):
		html += '<tr><td>  </td></tr>'
	for i in range(n):
		html += '<tr><td>'
		html += str(stack[n-1-i])
		html += '</td></tr>'
	html += "</table>>'''"
	return html

def draw_graph(adj, colors, cycles):
	global id
	s = Digraph('graph', format='jpg')
	#s.attr(compound='true')
	edges = set([])
	for u in range(len(adj)):
		s.node(str(u), style='filled', color=colors[u])
	for u in range(len(adj)):
		for v in adj[u]:
			if not (u, v) in edges and not (v, u) in edges: 
				edges.add((u, v))
				s.edge(str(u), str(v), color=colors.get((u,v), 'lightgray'))
	s.attr(label=r'\n\nCycle detection in a Directed Graph with DFS (Recursive)\n Number of cycles found: {}, cycles = {}'.format(len(cycles), [','.join(map(str, reversed(x))) for x in cycles]))
	s.attr(fontsize='15')
	s.attr(size = "507,835")
	s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1


def acyclic2(adj):
	#write your code here
	n = len(adj)
	#visited = [False]*n
	visited = [False]*n
	parents = [None]*n
	on_stack = [False]*n
	cycles = []
	colors = {v:'floralwhite' for v in range(len(adj))}
	#adj1 = deepcopy(adj)

	def dfs_visit(adj, u, cycles):
		visited[u] = True
		on_stack[u] = True
		colors[u] = 'lightgray'
		draw_graph(adj, colors, cycles) #adj1
		
		for v in adj[u]:
			if not visited[v]:
				parents[v] = u
				colors[u, v] = 'dimgray' #colors[v, u] = 
				#if u in adj[v]:
				#	adj[v].remove(u)
				dfs_visit(adj, v, cycles)
			elif on_stack[v]:
				x = u
				cycle = []
				while x != v:
					colors[x] = colors[parents[x]] = 'red'
					colors[x, parents[x]] = colors[parents[x], x] = 'red'
					cycle.append(x)
					x = parents[x]
				cycle = cycle + [v]
				colors[cycle[0], cycle[-1]] = colors[cycle[-1], cycle[0]] = 'red'
				print(cycle)				
				cycles.append(cycle)
		colors[u] = 'dimgray'				
		on_stack[u] = False
		draw_graph(adj, colors, cycles) #adj1
		
	for v in range(n):
		if not visited[v]:
			dfs_visit(adj, v, cycles)
	
	return int(len(cycles) > 0)


def acyclic(adj):
	#write your code here
	n = len(adj)
	#visited = [False]*n
	visited = [False]*n
	parents = [None]*n
	on_stack = [False]*n
	cycle = []
	
	def dfs_visit(adj, u, cycle):
		visited[u] = True
		on_stack[u] = True
		for v in adj[u]:
			if not visited[v]:
				parents[v] = u
				dfs_visit(adj, v, cycle)
			elif on_stack[v]:
				x = u
				while x != v:
					cycle.append(x)
					x = parents[x]
				cycle = [v] + cycle
				#print(cycle)
				
		on_stack[u] = False
		
	for v in range(n):
		if not visited[v]:
			dfs_visit(adj, v, cycle)
	
	return int(len(cycle) > 0)

if __name__ == '__main__':
    '''
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    n, m = data[0:2]
    data = data[2:]
    edges = list(zip(data[0:(2 * m):2], data[1:(2 * m):2]))
    adj = [[] for _ in range(n)]
    for (a, b) in edges:
        adj[a - 1].append(b - 1)
    print(acyclic(adj))
    '''
    n, p = 20, 0.3 #25
    adj = gen_ER_graph(n, p)
    #adj, adj1 = gen_cycle(n)
    #adj = gen_cycle(n)
    print(acyclic2(adj))
    #print(acyclic2(adj1))
