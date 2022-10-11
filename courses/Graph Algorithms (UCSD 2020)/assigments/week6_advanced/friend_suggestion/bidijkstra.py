#Uses python3

import sys
import queue
import matplotlib.pylab as plt

#from collections import defaultdict
#from heapq import *

# http://users.diag.uniroma1.it/challenge9/download.shtml

from graphviz import Digraph #Graph
id = 0
from random import random, randint

INF = float('Inf') #10**9

def gen_ER_graph(n, p):
	adj = [[] for _ in range(n)]
	cost = {}
	for u in range(n):
		for v in range(u+1, n):
			if random() < p:
				w = randint(5,45)
				if random() < 0.5:
					adj[u].append(v)
					cost[u,v] = w
				else:
					adj[v].append(u)
					cost[v,u] = w
	print(adj)
	return adj, cost
	
def draw_graph2(adj, cost, colors, d, dr, s, t, x, y, dist=INF, path=None, iter=None):
	global id
	plt.figure(figsize=(8,8))
	plt.grid()
	#for u in range(len(adj)):
	#	plt.plot(x[u], y[u], color=cols[u], marker='.', markersize=20)
	#plt.plot(x[s], y[s], color='red', marker='.', markersize=50)
	#plt.plot(x[t], y[t], color='red', marker='.', markersize=50)
	plt.scatter(x, y, c=[colors[u] for u in range(len(adj))], s=20)
	plt.scatter(x[s], y[s], color='red', s=50)
	plt.scatter(x[t], y[t], color='red', s=50)
	if path:
		for i in range(len(path)-1):
			u, v = path[i], path[i+1]
			plt.plot([x[u],x[v]], [y[u],y[v]], '.-', color='red')
	#for u in range(len(adj)):
	#	for v in adj[u]:
	#		if colors.get((u,v)):
	#			plt.plot([x[u],x[v]], [y[u],y[v]], '-', color=colors[u,v])
	plt.title('Shortest path with bidirectional Dijkstra between the red points' + (' (iteration {})'.format(iter) if iter != None else ''), size=12)
	plt.savefig('out1/plane_{:03d}.png'.format(id))
	plt.close()
	id += 1


def draw_graph(adj, cost, colors, d, dr, dist=INF, path=None):
	global id
	s = Digraph('graph', format='jpg')
	#s.attr(compound='true')
	edges = set([])
	for u in range(len(adj)):
		s.node(str(u), str(u) + '\n(' + str(d[u] if d[u] < INF else dr[u] if dr[u] < INF else '∞') + ')', style='filled', color=colors[u])
	for u in range(len(adj)):
		for v in adj[u]:
			if not (u, v) in edges: 
				edges.add((u, v))
				s.edge(str(u), str(v), color=colors.get((u,v), 'lightgray'), label=str(cost.get((u,v), '∞')))
	s.attr(label=r'\n\nShortest Distance to {:d} from {:d} with Dijkstra = {}'.format(y, x, dist if dist < INF else '∞'))
	s.attr(fontsize='25')
	s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1
	
def distance2(adj, cost, adjr, costr, s, t, x, y):
	#write your code here
		
	def process(u, adj, cost, h, d, prev, visited, colors, f=True):
		for v in adj[u]:
			if d[v] > d[u] + cost[u,v]:
				d[v] = d[u] + cost[u,v]
				h.put((d[v], v))
				prev[v] = u
				colors[u,v] = colors[v,u] = 'darkgreen' if f else 'dodgerblue'
				#print(u, v, d[v])
		#proc.add(u)
		visited[u] = True
	
	def shortest_path(s, dist, prev, t, distR, prevR, visited_any):
		distance = inf
		ubest = None
		for u in visited_any:
		#for u in proc.union(procR):
			if dist[u] + distR[u] < distance:
				ubest = u
				distance = dist[u] + distR[u]
		if ubest == None:
			return (-1, None)
		path = []
		last = ubest
		while last != s:
			path.append(last)
			last = prev[last]
		path.append(s)
		path = list(reversed(path))
		last = ubest
		while last != t:
			last = prevR[last]
			path.append(last)
		return (distance, path)
	
	inf = INF #10**19
	n = len(adj)
	#adjr, costr = reverse_graph(adj, cost)
	
	d, dr = [inf]*n, [inf]*n
	d[s] = 0
	dr[t] = 0
	visited, visitedr = [False]*n, [False]*n
	visited_any = set([])
	prev, prevr = [None]*n, [None]*n
	h = queue.PriorityQueue()
	h.put((d[s], s))
	hr = queue.PriorityQueue()
	hr.put((dr[t], t))
	colors = {v:'lightgray' for v in range(len(adj))}
	colors[s] = colors[t] = 'pink'
	draw_graph2(adj, cost, colors, d, dr, s, t, x, y)
	
	dist, path = -1, None
	iter = 0

	while True:
		u = h.get()[1]
		if visited[u]: continue
		process(u, adj, cost, h, d, prev, visited, colors)
		visited_any.add(u)
		colors[u] = 'dimgray' #'greenyellow' #'bisque1' 
		#draw_graph2(adj, cost, colors, d, dr, s, t, x, y, dist, path)
		if visitedr[u]:
			dist, path = shortest_path(s, d, prev, t, dr, prevr, visited_any)
			if path:
				for i in range(len(path)-1):
					u, v = path[i], path[i+1]
					colors[u] = colors[v] = colors[u, v] = 'red'
			draw_graph2(adj, cost, colors, d, dr, s, t, x, y, dist, path)
			return dist, path
		ur = hr.get()[1]
		if visitedr[ur]: continue
		process(ur, adjr, costr, hr, dr, prevr, visitedr, colors, False)
		visited_any.add(ur)
		colors[ur] = 'dimgray' #'lightskyblue1'
		if iter % 2500 == 0:
			print(iter)
			draw_graph2(adj, cost, colors,  d, dr, s, t, x, y, dist, iter=iter)
		if visited[ur]:
			dist, path = shortest_path(s, d, prev, t, dr, prevr, visited_any)
			if path:
				for i in range(len(path)-1):
					u, v = path[i], path[i+1]
					colors[u] = colors[v] = colors[u, v] = 'red'
			draw_graph2(adj, cost, colors, d, dr, s, t, x, y, dist, path)
			return dist, path
		if h.empty() or hr.empty():
			return (-1, None)
		iter += 1

def reverse_graph(adj, cost):
		n = len(adj)
		new_adj = [ [] for _ in range(n)]
		new_cost = {}
		for u in range(n):
			for v in adj[u]:
				new_adj[v].append(u)
				new_cost[v, u] = cost[u, v]
		return new_adj, new_cost
		
def distance(adj, cost, adjr, costr, s, t):
	#write your code here
		
	#def process(u, adj, cost, h, d, prev, proc):
	def process(u, adj, cost, h, d, prev, visited):
		for i in range(len(adj[u])):
			v = adj[u][i]
			if d[v] > d[u] + cost[u][i]:
				d[v] = d[u] + cost[u][i]
				h.put((d[v], v))
				prev[v] = u
		#proc.add(u)
		visited[u] = True
	
	#def shortest_path(s, dist, prev, proc, t, distR, prevR, procR):
	def shortest_path(s, dist, prev, t, distR, prevR, visited_any):
		distance = inf
		ubest = None
		for u in visited_any:
		#for u in proc.union(procR):
			if dist[u] + distR[u] < distance:
				ubest = u
				distance = dist[u] + distR[u]
		return distance if ubest != None else -1
	
	inf = 10**19
	n = len(adj)
	#adjr, costr = reverse_graph(adj, cost)
	
	d, dr = [inf]*n, [inf]*n
	d[s] = 0
	dr[t] = 0
	visited, visitedr = [False]*n, [False]*n
	visited_any = set([])
	prev, prevr = [None]*n, [None]*n
	#proc, procr = set([]), set([])
	h = queue.PriorityQueue()
	h.put((d[s], s))
	#for v in range(n):
	#	h.put((d[v], v))
	hr = queue.PriorityQueue()
	hr.put((dr[t], t))
	#for v in range(n):
	#	hr.put((dr[v], v))
	while True:
		u = h.get()[1]
		#if u in proc: continue
		#process(u, adj, cost, h, d, prev, proc)
		#if u in procr: 
		#	return shortest_path(s, d, prev, proc, t, dr, prevr, procr)
		if visited[u]: continue
		process(u, adj, cost, h, d, prev, visited)
		visited_any.add(u)
		if visitedr[u]:
			return shortest_path(s, d, prev, t, dr, prevr, visited_any)
		ur = hr.get()[1]
		#if ur in procr: continue
		#process(ur, adjr, costr, hr, dr, prevr, procr)
		#if ur in proc: #visited[ur]:
		#	return shortest_path(s, d, prev, proc, t, dr, prevr, procr)
		if visitedr[ur]: continue
		process(ur, adjr, costr, hr, dr, prevr, visitedr)
		visited_any.add(ur)
		if visited[ur]:
			return shortest_path(s, d, prev, t, dr, prevr, visited_any)
		if h.empty() or hr.empty():
			return -1 #(-1, None)
			
if __name__ == '__main__':
	'''
	input = sys.stdin.read()
	data = list(map(int, input.split()))
	n, m = data[0:2]
	edges = data[2:2+3*m]
	q = data[2+3*m]
	qs = data[2+3*m+1:]
	edges = list(zip(zip(edges[0:(3 * m):3], edges[1:(3 * m):3]), edges[2:(3 * m):3]))
	adj = [[] for _ in range(n)]
	cost = [[] for _ in range(n)]
	adjr = [[] for _ in range(n)]
	costr = [[] for _ in range(n)]
	for ((a, b), w) in edges:
		adj[a - 1].append(b - 1)
		cost[a - 1].append(w)
		adjr[b - 1].append(a - 1)
		costr[b - 1].append(w)
	#print(n, m, q)
	#print(adj)
	#print(cost)
	for u, v in (list(zip(qs[0:(2 * q):2], qs[1:(2 * q):2]))):
		#print(u, v)
		#d, p = distance(adj, cost, u-1, v-1)
		#print(d)
		#print(distance(adj, cost, adjr, costr, u-1, v-1))
		print(distance2(adj, cost, adjr, costr, u-1, v-1))
	'''
	#n, p = 20, 0.3
	#adj, w = gen_ER_graph(n, p)
	#adjr, wr = reverse_graph(adj, w)
	#s, t = randint(0, n-1), randint(0, n-1)
	#print(s, t)
	#d, p = distance2(adj, w, adjr, wr, s, t)
	#print(d, p)
	lines = open('USA-road-d.BAY.gr').readlines()
	adj, cost = None, {}
	adjr, costr = None, {}
	i = 0
	for line in lines:
		if line[0] == 'c': continue
		if line[0] == 'p':
			_, _, n, m = line.split()
			n, m = int(n), int(m)
			print(n, m)
			adj = [[] for _ in range(n)]
			adjr = [[] for _ in range(n)]
		if line[0] == 'a':
			_, u, v, w = line.split()
			u, v, w = map(int, (u, v, w))
			adj[u-1].append(v-1)
			adjr[v-1].append(u-1)
			cost[u-1,v-1] = min(w, cost.get((u-1,v-1),INF))
			costr[v-1,u-1] = min(w, cost.get((v-1,u-1),INF))
			i += 1
	#print(len(adj), len(cost), i)
	lines = open('USA-road-d.BAY.co').readlines()
	x, y = None, None
	i = 0
	for line in lines:
		if line[0] == 'c': continue
		if line[0] == 'p':
			n = int(line.split()[-1])
			print(n)
			x = [0]*n
			y = [0]*n
		if line[0] == 'v':
			_, v, x_, y_ = line.split()
			v, x_, y_ = map(int, (v, x_, y_))
			x[v-1] = x_
			y[v-1] = y_
			i += 1
	print(n, len(x), len(y), i)
	#s, t = 0, n-1
	#s = randint(0, n // 2)
	#t = randint(n // 2 + 1, n-1)
	yx = list(zip(y, x))
	#print(len(xy))
	s, t = min(yx), max(yx)
	s, t = yx.index(s), yx	.index(t)
	print(s, t)
	distance2(adj, cost, adjr, costr, s, t, x, y)