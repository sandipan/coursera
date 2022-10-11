#Uses python3

import sys

def dfs(adj, used, order, x):
    #write your code here
    pass

clock = 1

from graphviz import Digraph
id = 0
from random import random, randint, choice

def gen_ER_graph(n, p):
	adj = [[] for _ in range(n)]
	for u in range(n):
		for v in range(u+1, n):
			if random() < p:
				adj[u].append(v)
	print(adj)
	return adj
		
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

def draw_graph(adj, colors, previsit, postvisit, order=None):
	global id
	s = Digraph('graph', format='jpg')
	#s.attr(compound='true')
	edges = set([])
	for u in range(len(adj)):
		s.node(str(u), str(u) + '\n(' + str(previsit[u] if previsit[u] else '∞')  + '/' + str(postvisit[u] if postvisit[u] else '∞') + ')', style='filled', color=colors[u])
	for u in range(len(adj)):
		for v in adj[u]:
			if not (u, v) in edges and not (v, u) in edges: 
				edges.add((u, v))
				s.edge(str(u), str(v), color=colors.get((u,v), 'lightgray'))
	s.attr(label=r'\n\nTopological sort in a Directed Graph with DFS (Recursive), Nodes: {} \nTopological Order: {}'.format(','.join(map(str, range(len(adj)))), '→'.join(map(str, order)) if order else ''))
	s.attr(fontsize='15')
	s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1

def toposort2(adj):
	order = []
	#write your code here
	n = len(adj)
	visited = [False]*n
	previsit = [0]*n
	postvisit = [0]*n
	colors = {v:'floralwhite' for v in range(len(adj))}

	def dfs_visit(adj, u):
		global clock
		visited[u] = True
		previsit[u] = clock
		colors[u] = 'lightgray'
		draw_graph(adj, colors, previsit, postvisit)
		clock += 1
		for v in adj[u]:
			if not visited[v]:
				colors[u, v] = 'dimgray' 
				dfs_visit(adj, v)
		postvisit[u] = clock
		clock += 1
		colors[u] = 'dimgray'
	
	for v in range(n):
		if not visited[v]:
			dfs_visit(adj, v)
			draw_graph(adj, colors, previsit, postvisit)
	
	#for v in range(n):
	#	print(v, previsit[v], postvisit[v])
	order = [x for _, x in sorted(zip(postvisit, range(n)), key=lambda pair: pair[0], reverse=True)]
	draw_graph(adj, colors, previsit, postvisit, order)
	
	return order
	
def toposort(adj):
	order = []
	#write your code here
	n = len(adj)
	visited = [False]*n
	previsit = [0]*n
	postvisit = [0]*n
	
	def dfs_visit(adj, u):
		global clock
		visited[u] = True
		previsit[u] = clock
		clock += 1
		for v in adj[u]:
			if not visited[v]:
				dfs_visit(adj, v)
		postvisit[u] = clock
		clock += 1
	
	for v in range(n):
		if not visited[v]:
			dfs_visit(adj, v)
	
	#for v in range(n):
	#	print(v, previsit[v], postvisit[v])
	order = [x for _, x in sorted(zip(postvisit, range(n)), key=lambda pair: pair[0], reverse=True)]
	
	return order

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
    order = toposort(adj)
    for x in order:
        print(x + 1, end=' ')
    '''
    n, p = 20, 0.2 #25
    adj = gen_ER_graph(n, p)
    print(toposort2(adj))
