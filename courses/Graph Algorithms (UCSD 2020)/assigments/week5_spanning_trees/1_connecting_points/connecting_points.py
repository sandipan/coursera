#Uses python3
import sys
import math
import queue

from graphviz import Graph
id = 0
from random import random, randint, choice
import matplotlib.pylab as plt

INF = 10**9

def gen_points(n):
	points = set([])
	for i in range(n):
		x = randint(0,50)
		y = randint(0,50)
		points.add((x,y))
	return [x[0] for x in points], [x[1] for x in points]
	
def gen_points2(n):
	points = set([])
	for i in range(n):
		for j in range(n):
			points.add((i,j))
	return [x[0] for x in points], [x[1] for x in points]

def draw_graph(adj, cost, x, y, colors, m):
	global id
	#s = Graph('graph_bfs', format='jpg')
	#s.attr(compound='true')
	plt.figure(figsize=(5,5)) # (10,10)
	#edges = set([])
	for u in range(len(adj)):
		#s.node(str(u), str(u) + '\n(' + str('{:.02f}'.format(cost[u]) if cost[u] < INF else '∞') + ')', style='filled', color=colors[u])
		#plt.plot(x[u], y[u], '.r', size=10)
		plt.scatter(x[u], y[u], c='red', s=50)
	for u in range(len(adj)):
		for v in range(u+1, len(adj)):
			#if u == v: continue
			#edges.add((u, v))
			#s.edge(str(u), str(v), color=colors.get((u,v), 'lightgray'), label=str('{:.02f}'.format(adj[u][v])))
			if colors.get((u,v), 'lightgray') in ['dimgray', 'red']:
				plt.plot([x[u], x[v]], [y[u], y[v]], 'b-')
	plt.grid()
	plt.title('Minimum Cost Spanning Tree (MCST) with Prim (greedy), cost = ' + ('{:.02f}'.format(m) if m < INF else '∞'), size=10)
	plt.savefig('out1/graph_{:03d}'.format(id))
	plt.close()
	#s.attr(label='\n\nMinimum Cost Spanning Tree (MCST) with Prim (greedy), cost = ' + ('{:.02f}'.format(m) if m < INF else '∞'))
	#s.attr(fontsize='15')
	#s.edge('graph', 'q', color='white')
	#s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1
	
def minimum_distance2(x, y):
	result = 0.
    #write your code here
	inf = 10**19
	n = len(x)
	adj = [[0 for _ in range(n)] for _ in range(n)]
	for i in range(n):
		for j in range(i+1, n):
			adj[i][j] = adj[j][i] = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
	#print(adj)
	c = [inf]*n
	s = 0
	c[s] = 0
	colors = {v:'floralwhite' for v in range(len(adj))}
	colors[s] = 'pink'
	visited = [False]*n
	parent = [None]*n
	draw_graph(adj, c, x, y, colors, inf)
	h = queue.PriorityQueue()
	for v in range(n):
		h.put((c[v], v))
	while not h.empty():
		w, u = h.get()
		if visited[u]: continue
		visited[u] = True
		for v in range(n):
			if v == u: continue
			if (not visited[v]) and (c[v] > adj[u][v]):
				c[v] = adj[u][v]
				if parent[v] != None:
					colors[v, parent[v]] = colors[parent[v], v] = 'lightgray'
				parent[v] = u
				colors[v, u] = colors[u, v] = 'dimgray'
				h.put((c[v], v))
				draw_graph(adj, c, x, y, colors, inf)
	spanning_tree = []
	for i in range(n):
		colors[i] = colors[parent[i]] = 'red'
		colors[i, parent[i]] = colors[parent[i], i] = 'red'
		spanning_tree.append((i, parent[i]))
		if parent[i] != None: 
			result += adj[i][parent[i]]
	draw_graph(adj, c, x, y, colors, result)
	print(spanning_tree)
	return result
	
def minimum_distance(x, y):
	result = 0.
    #write your code here
	inf = 10**19
	n = len(x)
	adj = [[0 for _ in range(n)] for _ in range(n)]
	for i in range(n):
		for j in range(i+1, n):
			adj[i][j] = adj[j][i] = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
	#print(adj)
	c = [inf]*n
	s = 0
	c[s] = 0
	visited = [False]*n
	parent = [None]*n
	h = queue.PriorityQueue()
	for v in range(n):
		h.put((c[v], v))
	while not h.empty():
		w, u = h.get()
		if visited[u]: continue
		visited[u] = True
		for v in range(n):
			if v == u: continue
			if (not visited[v]) and (c[v] > adj[u][v]):
				c[v] = adj[u][v]
				parent[v] = u
				h.put((c[v], v))
	spanning_tree = []
	for i in range(n):
		spanning_tree.append((i, parent[i]))
		if parent[i] != None: 
			result += adj[i][parent[i]]
	#print(spanning_tree)
	return result

if __name__ == '__main__':
    '''
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    n = data[0]
    x = data[1::2]
    y = data[2::2]
    print("{0:.9f}".format(minimum_distance(x, y)))
    '''
    #x, y = gen_points(8)
    x, y = gen_points2(6)
    minimum_distance2(x, y)