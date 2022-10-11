#Uses python3

import sys
from graphviz import Graph
id = 0
from random import random, randint

def gen_ER_graph(n, p, c):
	ncs = [randint(2, n) for _ in range(c)]
	print(ncs)
	adj = [[] for _ in range(sum(ncs))]
	x = 0
	for i in range(c):
		nc = ncs[i]
		for u in range(x, x + nc):
			for v in range(u+1, x + nc):
				if random() < p:
					adj[u].append(v)
					adj[v].append(u)
		x += nc
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

def draw_graph(adj, colors, ncomp, stack=None):
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
	s.attr(label=r'\n\nConnected Components in an Undirected Graph with DFS, found={}'.format(ncomp))
	s.attr(fontsize='15')
	s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1
	
def number_of_components2(adj):
	result = 0
	#write your code here
	n = len(adj)
	colors = {v:'floralwhite' for v in range(len(adj))}
	draw_graph(adj, colors, result)
	
	ccolors = ['cyan2', 'darksalmon', 'greenyellow', 'indianred1', 'lightblue2', 'pink1']
	
	#visited = [False]*len(adj)
	for x in range(n):
		if colors[x] == 'floralwhite': #not visited[x]:
			result += 1
			stack = [x]
			colors[x] = 'lightgrey'
			draw_graph(adj, colors, result, stack)
			cc = [x]
			while len(stack) > 0:
				u = stack.pop(-1)
				draw_graph(adj, colors, result, stack)
				#visited[u] = True
				for v in adj[u]:
					#if not visited[v]:
					if colors[v] == 'floralwhite':
						stack.append(v)
						cc.append(v)
						colors[v] = 'lightgrey'
						colors[u, v] = colors[v, u] = 'dimgrey'
						draw_graph(adj, colors, result, stack)
				colors[u] = 'dimgrey'
				draw_graph(adj, colors, result, stack)
			for v in cc:
				colors[v] = ccolors[result-1]
			draw_graph(adj, colors, result, stack)
	return result

def number_of_components(adj):
	result = 0
	#write your code here
	n = len(adj)
	visited = [False]*len(adj)
	for x in range(n):
		if not visited[x]:
			result += 1
			stack = [x]
			while len(stack) > 0:
				u = stack.pop(-1)
				visited[u] = True
				for v in adj[u]:
					if not visited[v]:
						stack.append(v)	
	return result

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
    print(number_of_components(adj))
    '''
    n, p, c = 10, 0.275, 3
    adj = gen_ER_graph(n, p, c)
    print(number_of_components2(adj))
