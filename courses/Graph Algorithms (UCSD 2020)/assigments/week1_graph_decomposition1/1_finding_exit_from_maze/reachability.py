#Uses python3

import sys
from graphviz import Graph
id = 0
from random import random, randint

def gen_ER_graph(n, p):
	adj = [[] for _ in range(n)]
	for u in range(n):
		for v in range(u+1, n):
			if random() < p:
				adj[u].append(v)
				adj[v].append(u)
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

def draw_graph(adj, colors, x, y, stack=None):
	global id
	s = Graph('graph', format='jpg')
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
	#if stack:
	with s.subgraph(name='stack') as c:
		c.node(str(len(adj)+1), eval(get_stack(stack))) #, style='filled', color='mistyrose')
		c.attr(label='Stack')
		c.attr(fontsize='15')
	s.attr(label=r'\n\nReach {:d} from {:d} with DFS'.format(y, x))
	s.attr(fontsize='15')
	#s.edge('graph', 'stack', color='white')
	s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1
	
def reach2(adj, x, y):
	#write your code here
	colors = {v:'floralwhite' for v in range(len(adj))}
	colors[x] = colors[y] = 'pink'
	draw_graph(adj, colors, x, y)
	
	#visited = [False]*len(adj)
	stack = [x]
	colors[x] = 'lightgrey'
	draw_graph(adj, colors, x, y, stack)
	par = {}
	while len(stack) > 0:
		u = stack.pop(-1)
		draw_graph(adj, colors, x, y, stack)
		#visited[u] = True
		for v in adj[u]:
			#if not visited[v]:
			if colors[v] == 'floralwhite' or (v == y and colors[y] == 'pink'):
				stack.append(v)
				par[v] = u
				colors[v] = 'lightgrey'
				colors[u, v] = colors[v, u] = 'dimgrey'
				if v == y:
					colors[x] = colors[y] = 'pink'
					while v != x:
						colors[v, par[v]] = colors[par[v], v] ='red'
						v = par[v]
						colors[v] = 'pink'
					draw_graph(adj, colors, x, y, stack)
					return 1
				draw_graph(adj, colors, x, y, stack)
		colors[u] = 'dimgrey'
		draw_graph(adj, colors, x, y, stack)
	colors[x] = colors[y] = 'pink'
	draw_graph(adj, colors, x, y, stack)
	return 0


def reach(adj, x, y):
	#write your code here
	visited = [False]*len(adj)
	stack = [x]
	while len(stack) > 0:
		u = stack.pop(-1)
		if u == y:
			return 1
		visited[u] = True
		for v in adj[u]:
			if not visited[v]:
				stack.append(v)			

	return 0

if __name__ == '__main__':
    '''
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    n, m = data[0:2]
    data = data[2:]
    edges = list(zip(data[0:(2 * m):2], data[1:(2 * m):2]))
    x, y = data[2 * m:]
    adj = [[] for _ in range(n)]
    x, y = x - 1, y - 1
    for (a, b) in edges:
        adj[a - 1].append(b - 1)
        adj[b - 1].append(a - 1)
    print(reach(adj, x, y))
    '''
    n, p = 20, 0.125
    adj = gen_ER_graph(n, p)
    x, y = randint(0, n-1), randint(0, n-1)
    print(x, y)
    print(reach2(adj, x, y))
