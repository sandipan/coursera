#Uses python3

import sys
sys.setrecursionlimit(200000)

from graphviz import Digraph
from random import random, randint
id = 0
clock = 1

def gen_ER_graph(n, p, c):
	ncs = [randint(2, n) for _ in range(c)]
	print(ncs)
	adj = [[] for _ in range(sum(ncs))]
	crs = [None for _ in range(c)]
	x = 0
	for i in range(c):
		nc = ncs[i]
		for u in range(x, x + nc):
			for v in range(u+1, x + nc):
				if random() < p:
					if random() < 0.75:
						adj[u].append(v)
					else:
						adj[v].append(u)
		crs[i] = randint(x, x+nc-1)
		x += nc
	
	for i in range(c-1):
		if random() < 0.5:
			adj[crs[i]].append(crs[i+1])
		else:
			adj[crs[i+1]].append(crs[i])
	
	print(adj)
	return adj
	
def draw_graph(adj, colors, previsit, postvisit, ncomp, txt=''):
	global id
	s = Digraph('graph', format='jpg')
	edges = set([])
	for u in range(len(adj)):
		s.node(str(u), str(u) + '\n(' + str(previsit[u] if previsit[u] else '∞')  + '/' + str(postvisit[u] if postvisit[u] else '∞') + ')', style='filled', color=colors[u])
	for u in range(len(adj)):
		for v in adj[u]:
			if not (u, v) in edges: 
				edges.add((u, v))
				s.edge(str(u), str(v), color=colors.get((u,v), 'lightgray'))
	s.attr(label=r'\n\nStrongly Connected Components (SCC) in a Directed Graph with DFS (KosaRaju), found={}\n{}'.format(ncomp, txt))
	s.attr(fontsize='15')
	s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1
	
def number_of_strongly_connected_components2(adj):
	result = 0
	#write your code here
	n = len(adj)
	visited = [False]*n
	previsit = [0]*n
	postvisit = [0]*n
	colors = {v:'floralwhite' for v in range(n)}
	draw_graph(adj, colors, previsit, postvisit, result)
	
	ccolors = ['cyan2', 'gold1', 'darkolivegreen1', 'darksalmon', 'greenyellow', 'indianred1', 'lightblue2', 'pink1', 'orchid', 'powderblue', 'yellow', 'tan', 'sandybrown', 'beige', 'aquamarine', 'lavenderblush']
	
	def reverse_graph(adj):
		n = len(adj)
		new_adj = [ [] for _ in range(n)]
		for i in range(n):
			for j in adj[i]:
				new_adj[j].append(i)
		return new_adj
	
	def dfs_visit(adj, u, colors, cc=None, txt=''):
		global clock
		visited[u] = True
		previsit[u] = clock
		colors[u] = 'lightgrey'
		draw_graph(adj, colors, previsit, postvisit, result, txt)
		clock += 1
		if cc != None:
			cc.append(u)
		for v in adj[u]:
			if not visited[v]:
				#cc.append(v)
				colors[v] = 'lightgrey'
				colors[u, v] = colors[v,u] = 'dimgrey'
				dfs_visit(adj, v, colors, cc, txt)
		postvisit[u] = clock
		colors[u] = 'dimgrey'
		draw_graph(adj, colors, previsit, postvisit, result, txt)
		clock += 1
	
	for v in range(n):
		if not visited[v]:
			dfs_visit(adj, v, colors, txt='DFS to find timstamps')
	post_v = [x for _, x in sorted(zip(postvisit, range(n)), key=lambda pair: pair[0], reverse=True)]
	print(post_v)
	rev_adj = reverse_graph(adj)
	#for v in range(n):
	#	print(v, previsit[v], postvisit[v])
	#print(post_v)
	
	colors = {v:'floralwhite' for v in range(n)}
	visited = [False]*n
	for v in post_v:
		cc = []
		if not visited[v]:
			dfs_visit(rev_adj, v, colors, cc, 'DFS on Reverse (Transpose) Graph to find CC')
			result += 1
		for v in cc:
			colors[v] = ccolors[(result-1) % len(ccolors)]
	draw_graph(rev_adj, colors, previsit, postvisit, result, 'DFS on Reverse (Transpose) Graph to find CC')
	
	for u in range(n):
		for v in adj[u]:
			colors[u, v] = colors[v,u] = 'dimgrey'
	
	draw_graph(adj, colors, previsit, postvisit, result, 'CC in the Original Graph')
	
	return result
	
def number_of_strongly_connected_components(adj):
	result = 0
	#write your code here
	visited = [False]*n
	previsit = [0]*n
	postvisit = [0]*n
	
	def reverse_graph(adj):
		n = len(adj)
		new_adj = [ [] for _ in range(n)]
		for i in range(n):
			for j in adj[i]:
				new_adj[j].append(i)
		return new_adj
	
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
	post_v = [x for _, x in sorted(zip(postvisit, range(n)), key=lambda pair: pair[0], reverse=True)]
	rev_adj = reverse_graph(adj)
	#for v in range(n):
	#	print(v, previsit[v], postvisit[v])
	#print(post_v)
	visited = [False]*n
	for v in post_v:
		if not visited[v]:
			dfs_visit(rev_adj, v)
			result += 1

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
    print(number_of_strongly_connected_components(adj))
    '''
    n, p, c = 10, 0.75, 2
    adj = gen_ER_graph(n, p, c)
    print(number_of_strongly_connected_components2(adj))
