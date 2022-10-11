#Uses python3

import sys
import queue

from graphviz import Graph
id = 0
from random import random, randint, choice

INF = 10**9

def gen_ER_graph(n, p):
	adj = [[] for _ in range(n)]
	for u in range(n):
		for v in range(u+1, n):
			if random() < p:
				adj[u].append(v)
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
		
def get_queue(q):
	html = "'''<<table>"
	n =  len(q) if q else 0
	html += '<tr>'
	for i in range(n):
		html += '<td>'
		html += str(q[i])
		html += '</td>'
	for i in range(15-n):
		html += '<td>&nbsp;</td>'
	html += '</tr>'
	html += "</table>>'''"
	return html

def draw_graph(adj, colors, q=None, result=None):
	global id
	s = Graph('graph_bfs', format='jpg')
	#s.attr(compound='true')
	with s.subgraph(name='graph') as c:
		edges = set([])
		for u in range(len(adj)):
			c.node(str(u), style='filled', color=colors[u])
		for u in range(len(adj)):
			for v in adj[u]:
				if not (u, v) in edges and not (v, u) in edges: 
					edges.add((u, v))
					c.edge(str(u), str(v), color=colors.get((u,v), 'lightgray'))
		c.attr(label='Graph')
		c.attr(fontsize='15')
	#if q:
	with s.subgraph(name='queue') as c:
		c.node(str(len(adj)+1), eval(get_queue(q))) #, style='filled', color='mistyrose')
		c.attr(label='q')
		c.attr(fontsize='15')
	s.attr(label='\n\nCheck if a Graph is bipartite (2-colorable) with BFS' + ('' if result == None else ': Bipartite' if result == 1 else ': Not Bipartite'))
	s.attr(fontsize='15')
	#s.edge('graph', 'q', color='white')
	s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1
	
def bipartite2(adj):
    #write your code here
	color = {i:'floralwhite' for i in range(len(adj))}
	draw_graph(adj, color)
	for vertex in range(len(adj)):
		if color[vertex] == 'floralwhite':
			queue = [vertex]
			color[vertex] = 'red'
			draw_graph(adj, color, queue)
			while len(queue) > 0:
				u = queue.pop(0)
				for v in adj[u]:
					if color[v] ==  color[u]:
						color[u,v] = color[v,u] = 'red'
						draw_graph(adj, color, queue, 0)
						return 0
					if color[v] == 'floralwhite':
						color[u,v] = color[v,u] = 'dimgrey'
						queue.append(v)
						color[v] = 'red' if color[u] == 'blue' else 'blue'
						draw_graph(adj, color, queue)

	draw_graph(adj, color, queue, 1)
	return 1
	
def bipartite(adj):
    #write your code here
	color = [None]*len(adj)
	for vertex in range(len(adj)):
		if not color[vertex]:
			queue = [vertex]
			color[vertex] = 'red'
			while len(queue) > 0:
				u = queue.pop(0)
				for v in adj[u]:
					if color[v] ==  color[u]:
						return 0
					if not color[v]:
						queue.append(v)
						color[v] = 'red' if color[u] == 'blue' else 'blue'
	return 1
	
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
        adj[b - 1].append(a - 1)
    print(bipartite(adj))
    '''
    n, p = 15, 0.1
    #adj = gen_ER_graph(n, p)
    adj = gen_cycle(n)
    print(bipartite2(adj))
    