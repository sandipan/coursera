#!/usr/bin/python3

import sys
import queue
import math

from graphviz import Graph #Digraph
id = 0
from random import random, randint
INF = float('Inf')
from matplotlib import pylab as plt

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
	
def draw_graph(adj, cost, colors, x, y, d, p, v1, v2, inf, path=None, iter=None):
	global id
	#s = Graph('graph', format='jpg')
	#s.attr(compound='true')
	#edges = set([])
	#for u in range(len(adj)):
	#	s.node(str(u), str(u) + '\n(' + str(d[u] if d[u] < inf else '∞') + ',' + str(round(p[u],2) if p.get(u, None) else '∞') + ')', style='filled', color=colors[u])
	#for u in range(len(adj)):
	#	for v, w in zip(adj[u], cost[u]):
	#		if not (u, v) in edges and not (v, u) in edges: 
	#			edges.add((u, v))
	#			s.edge(str(u), str(v), color=colors.get((u,v), 'lightgray'), label=str(w))
	#s.attr(label=r'\n\nShortest Distance to {:d} from {:d} with A* = {}'.format(v2, v1, d[v2] if d[v2] < inf else '∞'))
	#s.attr(fontsize='15')
	#s.render('out/graph_{:03d}'.format(id), view=False)
	plt.figure(figsize=(8,8))
	plt.grid()
	plt.scatter(x, y, c=[colors[u] for u in range(len(adj))], s=20)
	plt.scatter(x[v1], y[v1], color='red', s=50)
	plt.scatter(x[v2], y[v2], color='red', s=50)
	if path:
		for i in range(len(path)-1):
			u, v = path[i], path[i+1]
			plt.plot([x[u],x[v]], [y[u],y[v]], '.-', color='red')
	#for u in range(len(adj)):
	#	for v in adj[u]:
	#		if colors.get((u,v)):
	#			plt.plot([x[u],x[v]], [y[u],y[v]], '-', color=c
	plt.title('Shortest path with A* between the red points' + (' (iteration {})'.format(iter) if iter != None else ''), size=15)
	plt.savefig('out1/plane_{:03d}.png'.format(id))
	plt.close()
	id += 1
	
class AStar:
	def __init__(self, n, adj, cost, x, y):
		# See the explanations of these fields in the starter for friend_suggestion        
		self.n = n;
		self.adj = adj
		self.cost = cost
		self.inf = INF #n*10**6
		self.d = [self.inf]*n
		self.p = {}
		self.visited = [False]*n
		self.workset = []
		# Coordinates of the nodes
		self.x = x
		self.y = y

	# See the explanation of this method in the starter for friend_suggestion
	def clear(self):
		for v in self.workset:
			self.d[v] = self.inf
			self.visited[v] = False;
		del self.workset[0:len(self.workset)]

	# See the explanation of this method in the starter for friend_suggestion
	def visit(self, q, v, s, t, dist, measure):
		# Implement this method yourself
		self.d[v] = dist
		q.put((self.d[v] + measure(v, t) - measure(s, t), v))
		self.workset.append(v)
		
	def potential(self, v, t):
		if v not in self.p:
			self.p[v] = math.sqrt((self.x[v]-self.x[t])**2+(self.y[v]-self.y[t])**2)
		return self.p[v]

	def process(self, q, u, t, colors, par):
		#for v, w in zip(self.adj[u], self.cost[u]):
		for v in self.adj[u]:
			w = cost[u,v]
			if self.d[v] > self.d[u] + w:
				self.visit(q, v, s, t, 	self.d[u] + w, self.potential)
				colors[u, v] = 'dimgray'
				par[v] = u

	# Returns the distance from s to t in the graph
	def query(self, s, t):
		self.clear()
		q = queue.PriorityQueue()
		# Implement the rest of the algorithm yourself		
		n = len(self.adj)
		d = [self.inf]*n
		self.visit(q, s, s, t, 0, self.potential)
		colors = {v:'lightgray' for v in range(len(adj))}
		colors[s] = colors[t] = 'red'
		draw_graph(self.adj, self.cost, colors, self.x, self.y, self.d, self.p, s, t, self.inf)
		par = {}
		iter = 0
		while not q.empty():
			u = q.get()[1]
			if self.visited[u]: continue
			self.visited[u] = True
			colors[u] = 'dimgray'
			#print(u, self.p)
			if u == t:
				path = []
				while u != s:
					path.append(u)
					colors[u] = colors[par[u]] = colors[par[u],u] = colors[u, par[u]] = 'red'
					u = par[u]
				colors[s] = 'red'
				path.append(s)
				path = list(reversed(path))
				#print(path)
				draw_graph(self.adj, self.cost, colors, self.x, self.y, self.d, self.p, s, t, self.inf, path)
				return (self.d[t] if self.d[t] != self.inf else -1)
			self.process(q, u, t, colors, par)
			if iter % 2500 == 0:
				print(iter)
				draw_graph(self.adj, self.cost, colors, self.x, self.y, self.d, self.p, s, t, self.inf, iter=iter)
			iter += 1
			#self.workset.remove(u)
		
		return -1
		
class AStar1:
	def __init__(self, n, adj, cost, x, y):
		# See the explanations of these fields in the starter for friend_suggestion        
		self.n = n;
		self.adj = adj
		self.cost = cost
		self.inf = n*10**6
		self.d = [self.inf]*n
		self.visited = [False]*n
		self.workset = []
		self.p = {}
		# Coordinates of the nodes
		self.x = x
		self.y = y

	def clear(self):
		for v in self.workset:
			self.d[v] = self.inf
			self.visited[v] = False;
		del self.workset[0:len(self.workset)]
		self.p = {}

	def visit(self, q, v, dist, measure):
		# Implement this method yourself
		if self.d[v] > dist:
			self.d[v] = dist
			q.put((self.d[v] + measure, v))
			self.workset.append(v)

	def potential(self, u, t):
		if u not in self.p:
			u = (self.x[u], self.y[u])
			t = (self.x[t], self.y[t])
			self.p[u] = math.sqrt((u[0]-t[0])**2+(u[1]-t[1])**2)
		return self.p[u]

	def extract_min(self, q):
		_, v = q.get()
		return v

	def process(self, q, v, t):
		#for u, w in zip(adj[v], cost[v]):
		for u in self.adj[v]:
			w = self.cost[v, u]
			if not self.visited[u]:
				self.visit(q, u, self.d[v] + w, self.potential(u, t))
			
	def query(self, s, t):
		self.clear()
		q = queue.PriorityQueue()
		self.visit(q, s, 0, self.potential(s, t))
		colors = {v:'lightgray' for v in range(len(adj))}
		colors[s] = colors[t] = 'red'
		draw_graph(self.adj, self.cost, colors, self.x, self.y, self.d, self.p, s, t, self.inf)
		par = {}
		iter = 0
		while not q.empty():
			v = self.extract_min(q)
			if v == t:
				path = []
				while v != s:
					path.append(v)
					colors[v] = colors[par[v]] = colors[par[v],v] = colors[v, par[v]] = 'red'
					v = par[v]
				colors[s] = 'red'
				path.append(s)
				path = list(reversed(path))
				#print(path)
				draw_graph(self.adj, self.cost, colors, self.x, self.y, self.d, self.p, s, t, self.inf)
				return (self.d[t] if self.d[t] != self.inf else -1)
			if not self.visited[v]:
				self.process(q, v, t)
				self.visited[v] = True
				colors[v] = 'blue'
			if iter % 1000 == 0:
				draw_graph(self.adj, self.cost, colors, self.x, self.y, self.d, self.p, s, t, self.inf)
			iter += 1
		return -1

def readl():
    return map(int, sys.stdin.readline().split())

if __name__ == '__main__':
	'''
	n,m = readl()
	x = [0 for _ in range(n)]
	y = [0 for _ in range(n)]
	adj = [[] for _ in range(n)]
	cost = [[] for _ in range(n)]
	for i in range(n):
		a, b = readl()
		x[i] = a
		y[i] = b
	for e in range(m):
		u,v,c = readl()
		adj[u-1].append(v-1)
		cost[u-1].append(c)
	t, = readl()
	astar = AStar(n, adj, cost, x, y)
	for i in range(t):
		s, t = readl()
		print(astar.query(s-1, t-1))
	'''

	lines = open('USA-road-d.BAY.gr').readlines()
	adj, cost = None, {}
	i = 0
	for line in lines:
		if line[0] == 'c': continue
		if line[0] == 'p':
			_, _, n, m = line.split()
			n, m = int(n), int(m)
			print(n, m)
			adj = [[] for _ in range(n)]
		if line[0] == 'a':
			_, u, v, w = line.split()
			u, v, w = map(int, (u, v, w))
			adj[u-1].append(v-1)
			cost[u-1,v-1] = min(w, cost.get((u-1,v-1),INF))
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
	astar = AStar(n, adj, cost, x, y)
	print(astar.query(s-1, t-1))
