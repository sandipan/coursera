#Uses python3

import sys
import queue

from graphviz import Graph
id = 0
from random import random, randint

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

def draw_graph(adj, colors, x, y, d, q=None):
	global id
	s = Graph('graph_bfs', format='jpg')
	#s.attr(compound='true')
	with s.subgraph(name='graph') as c:
		edges = set([])
		for u in range(len(adj)):
			c.node(str(u), str(u) + '\n(' + str(d[u] if d[u] < INF else '∞') + ')', style='filled', color=colors[u])
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
	s.attr(label=r'\n\nDistance to {:d} from {:d} with BFS = {}'.format(y, x, d[y] if d[y] < INF else '∞'))
	s.attr(fontsize='15')
	#s.edge('graph', 'q', color='white')
	s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1
	
def distance2(adj, s, t):
	#write your code here
	inf = INF
	colors = {v:'floralwhite' for v in range(len(adj))}
	colors[s] = colors[t] = 'pink'
	d = [inf]*len(adj)
	draw_graph(adj, colors, s, t, d)
	queue = [s]
	d[s] = 0
	par = {}
	colors[s] = 'lightgrey'
	draw_graph(adj, colors, s, t, d, queue)
	while len(queue) > 0:
		u = queue.pop(0)
		for v in adj[u]:
			if d[v] ==  inf:
				queue.append(v)
				par[v] = u
				d[v] = d[u] + 1
				colors[v] = 'lightgrey'
				colors[u, v] = colors[v, u] = 'dimgrey'
				draw_graph(adj, colors, s, t, d, queue)
				if v == t:
					colors[s] = colors[t] = 'pink'
					while v != s:
						colors[v, par[v]] = colors[par[v], v] ='red'
						v = par[v]
						colors[v] = 'pink'
					draw_graph(adj, colors, s, t, d, queue)
					return d[t]
		colors[u] = 'dimgrey'
	draw_graph(adj, colors, s, t, d, queue)
	return -1

def distance(adj, s, t):
	#write your code here
	inf = 1000000
	d = [inf]*len(adj)
	queue = [s]
	d[s] = 0
	while len(queue) > 0:
		u = queue.pop(0)
		for v in adj[u]:
			if d[v] ==  inf:
				queue.append(v)
				d[v] = d[u] + 1
				if v == t:
					return d[t]
	return -1

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
    s, t = data[2 * m] - 1, data[2 * m + 1] - 1
    print(distance(adj, s, t))
    '''
    n, p = 20, 0.1
    adj = gen_ER_graph(n, p)
    s, t = randint(0, n-1), randint(0, n-1)
    print(s, t)
    print(distance2(adj, s, t))
    