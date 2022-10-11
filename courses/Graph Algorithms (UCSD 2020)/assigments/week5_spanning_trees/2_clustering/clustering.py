#Uses python3
import sys
import math

from graphviz import Graph
id = 0
from random import random, randint, choice
from numpy.random import normal
import matplotlib.pylab as plt

import pandas as pd

INF = 10**9

def gen_points(n, k):
	C = [(100,200), (1000,500), (490,745), (150,950), (875, 555)]
	points = set([])
	for j in range(k):
		for i in range(n):
			x = normal(C[j][0], 250*random()) #+ randint(-20,20)
			y = normal(C[j][1], 250*random()) #+ randint(-20,20)
			points.add((x,y))
	return [x[0] for x in points], [x[1] for x in points]
	
def gen_points2(n):
	points = set([])
	for i in range(n):
		for j in range(n):
			points.add((i,j))
	return [x[0] for x in points], [x[1] for x in points]

def draw_graph(adj, x, y, colors, clusters, m):
	global id
	#s = Graph('graph_bfs', format='jpg')
	#s.attr(compound='true')
	plt.figure(figsize=(5,5)) # (10,10)
	#edges = set([])
	#for u in range(len(adj)):
		#s.node(str(u), style='filled', color=colors[u])
		#print(x[u], y[u], u, clusters[u])
	plt.scatter(x, y, c=[100*clusters[u] for u in clusters], cmap="plasma", s=5) #50)
	for u in range(len(adj)):
		for v in range(u+1, len(adj)):
			#if u == v: continue
			#edges.add((u, v))
			#s.edge(str(u), str(v), color=colors.get((u,v), 'lightgray'), label=str('{:.02f}'.format(adj[u][v])))
			if colors.get((u,v), 'lightgray') in ['dimgray', 'red']:
				plt.plot([x[u], x[v]], [y[u], y[v]], 'b-')#, color=[0.5,0.4,0.3])
	plt.grid()
	plt.title('Clustering with Kruskal (UnionFind), cost = ' + ('{:.02f}'.format(m) if m < INF else '∞'), size=10)
	plt.savefig('out1/graph_{:03d}'.format(id))
	plt.close()
	#s.attr(label='\n\nMinimum Cost Spanning Tree with Kruskal (greedy), cost = ' + ('{:.02f}'.format(m) if m < INF else '∞'))
	#s.attr(fontsize='15')
	#s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1

def clustering2(x, y, k): # path compression to be implemented
	#write your code here
	result = 0.
    #write your code here
	inf = float('Inf') #10**19
	n = len(x)
	adj = []
	cost = [[0]*n for _ in range(n)]
	for i in range(n):
		for j in range(i+1, n):
			d = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
			cost[i][j] = d
			adj.append((d, i, j))
	adj = sorted(adj)
	colors = {v:'floralwhite' for v in range(n)}
	ccolors = ['cyan2', 'gold1', 'darkolivegreen1', 'darksalmon', 'greenyellow', 'indianred1', 'lightblue2', 'pink1', 'orchid', 'powderblue', 'yellow', 'tan', 'sandybrown', 'beige', 'aquamarine', 'lavenderblush', 'cornflowerblue', 'chocolate1']
	ccolors += ['aliceblue', 'azure2', 'blanchedalmond', 'brown', 'burlywood', 'burlywood1', 'chartreuse', 'darkseagreen', 'deepskyblue', 'deeppink1', 'hotpink', 'ivory3', 'maroon1', 'rosybrown1']
	#colors = {v:ccolors[v] for v in range(n)}
	#print(adj)
	indices = [i for i in range(n)]
	draw_graph(cost, x, y, colors, indices, inf)
	spanning_tree =[]
	
	while len(set(indices)) > k:
		d, u, v = adj.pop(0)
		iu, iv = indices[u], indices[v]
		if iu != iv:
			indices[u] = indices[v] = min(iu, iv)		
			#colors[u] = colors[v] = min(colors[u], colors[v])
			for j in range(n):
				if indices[j] == max(iu, iv):
					indices[j] = min(iu, iv)
					#colors[j] = min(colors[u], colors[v])
			spanning_tree.append((u, v))
			colors[u, v] = colors[v, u] = 'dimgray'
			print(u, v, (x[u], y[u]), (x[v], y[v]), indices)
			draw_graph(cost, x, y, colors, indices, inf)
	#print((x[u], y[u]), (x[v], y[v]), d)
	clusters = {}
	for i in range(n):
		ci = indices[i]
		clusters[ci] = clusters.get(ci, []) + [i]
	#print(clusters)
	d = inf
	for i in list(clusters.keys()):
		for j in list(clusters.keys()):
			if i == j: continue
			for vi in clusters[i]:
				for vj in clusters[j]:
					d = min(d, math.sqrt((x[vi]-x[vj])**2 + (y[vi]-y[vj])**2))
	m = 0
	for (u, v) in spanning_tree:
		#colors[u] = colors[v] = 'red'
		#colors[u, v] = colors[v, u] = 'red'
		m += cost[u][v]		
	draw_graph(cost, x, y, colors, indices, m)
	
		
		
	return d

	
def clustering(x, y, k):
	#write your code here
	result = 0.
    #write your code here
	inf = float('Inf') #10**19
	n = len(x)
	adj = []
	for i in range(n):
		for j in range(i+1, n):
			adj.append((math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2), i, j))
	adj = sorted(adj)
	#print(adj)
	indices = {i:i for i in range(n)}
	while len(set(indices.values())) > k:
		d, u, v = adj.pop(0)
		iu, iv = indices[u], indices[v]
		if iu != iv:
			indices[u] = indices[v] = min(iu, iv)		
	#print((x[u], y[u]), (x[v], y[v]), d)
	clusters = {}
	for i in range(n):
		ci = indices[i]
		clusters[ci] = clusters.get(ci, []) + [i]
	#print(clusters)
	d = inf
	for i in list(clusters.keys()):
		for j in list(clusters.keys()):
			if i == j: continue
			for vi in clusters[i]:
				for vj in clusters[j]:
					d = min(d, math.sqrt((x[vi]-x[vj])**2 + (y[vi]-y[vj])**2))
	return d


if __name__ == '__main__':
    '''
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    n = data[0]
    data = data[1:]
    x = data[0:2 * n:2]
    y = data[1:2 * n:2]
    data = data[2 * n:]
    k = data[0]
    print("{0:.9f}".format(clustering(x, y, k)))
    '''
    #n, k = 50, 5
    #x, y = gen_points(n, k)
    df = pd.read_csv('Iris.csv')
    x = df. iloc[:,1].tolist()
    y = df. iloc[:,2].tolist()
    clustering2(x, y, 1)