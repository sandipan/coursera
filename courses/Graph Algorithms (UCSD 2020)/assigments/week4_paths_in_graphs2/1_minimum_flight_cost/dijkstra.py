#Uses python3

import sys
import queue

#from collections import defaultdict
#from heapq import *

from graphviz import Digraph #Graph
id = 0
from random import random, randint
import matplotlib.pylab as plt 

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

def draw_graph(adj, cost, colors, x, y, d):
	global id
	s = Digraph('graph', format='jpg')
	#s.attr(compound='true')
	edges = set([])
	for u in range(len(adj)):
		s.node(str(u), str(u) + '\n(' + str(d[u] if d[u] < INF else '∞') + ')', style='filled', color=colors[u])
	for u in range(len(adj)):
		for v in adj[u]:
			if not (u, v) in edges and not (v, u) in edges: 
				edges.add((u, v))
				s.edge(str(u), str(v), color=colors.get((u,v), 'lightgray'), label=str(cost.get((u,v), '∞')))
	s.attr(label=r'\n\nShortest Distance to {:d} from {:d} with Dijkstra = {}'.format(y, x, d[y] if d[y] < INF else '∞'))
	s.attr(fontsize='15')
	s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1
	
def draw_graph2(adj, cost, colors, x, y, s, t, d, path=None, iter=None):
	global id
	plt.figure(figsize=(8,8))
	plt.grid()
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
	#			plt.plot([x[u],x[v]], [y[u],y[v]], '-', color=c
	plt.title('Shortest path with Dijkstra between the red points' + (' (iteration {})'.format(iter) if iter != None else ''), size=15)
	plt.savefig('out1/plane_{:03d}.png'.format(id))
	plt.close()
	id += 1


'''
def dijkstra(edges, f, t):
    g = defaultdict(list)
    for l,r,c in edges:
        g[l].append((c,r))

    q, seen, mins = [(0,f,())], set(), {f: 0}
    while q:
        (cost,v1,path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == t: return (cost, path)

            for c, v2 in g.get(v1, ()):
                if v2 in seen: continue
                prev = mins.get(v2, None)
                next = cost + c
                if prev is None or next < prev:
                    mins[v2] = next
                    heappush(q, (next, v2, path))
'''

def distance2(adj, cost, s, t, x, y):
	#write your code here
	inf = INF
	n = len(adj)
	d = [inf]*n
	d[s] = 0
	visited = [0]*n
	h = queue.PriorityQueue()
	h.put((d[s], s))
	#for v in range(n):
	#	h.put((d[v], v))
	colors = {v:'lightgray' for v in range(len(adj))}
	colors[s] = colors[t] = 'pink'
	draw_graph2(adj, cost, colors, x, y, s, t, d)
	par = {}
	iter = 0
	while not h.empty():
		u = h.get()[1]
		if visited[u]: continue
		visited[u] = True
		colors[u] = 'dimgray'
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
			draw_graph2(adj, cost, colors, x, y, s, t, d, path)
			return (d[t] if d[t] != inf else -1)
		for v in adj[u]:
			if d[v] > d[u] + cost[u,v]:
				d[v] = d[u] + cost[u,v]
				par[v] = u
				colors[u, v] = colors[v, u] = 'blue'
				h.put((d[v], v))
				#draw_graph2(adj, cost, colors, x, y, s, t, d)
		if iter % 2500 == 0:
			print(iter)
			draw_graph2(adj, cost, colors, x, y, s, t, d, iter=iter)
		iter += 1
	colors[s] = colors[t] = 'pink'
	if d[t] < inf:
		v = t
		while v != s:
			colors[v, par[v]] = colors[par[v], v] ='red'
			v = par[v]
	#draw_graph2(adj, cost, colors, x, y, s, t, d)
	return d[t] if d[t] != inf else -1

def distance(adj, cost, s, t):
	#write your code here
	inf = 10**19
	n = len(adj)
	d = [inf]*n
	d[s] = 0
	visited = [0]*n
	h = queue.PriorityQueue()
	for v in range(n):
		h.put((d[v], v))
	while not h.empty():
		u = h.get()[1]
		if visited[u]: continue
		visited[u] = True
		for i in range(len(adj[u])):
			v = adj[u][i]
			if d[v] > d[u] + cost[u][i]:
				d[v] = d[u] + cost[u][i]
				h.put((d[v], v))
	return d[t] if d[t] != inf else -1


if __name__ == '__main__':
	'''
	input = sys.stdin.read()
	data = list(map(int, input.split()))
	n, m = data[0:2]
	data = data[2:]
	edges = list(zip(zip(data[0:(3 * m):3], data[1:(3 * m):3]), data[2:(3 * m):3]))
	data = data[3 * m:]
	adj = [[] for _ in range(n)]
	cost = [[] for _ in range(n)]
	for ((a, b), w) in edges:
		adj[a - 1].append(b - 1)
		cost[a - 1].append(w)
	s, t = data[0] - 1, data[1] - 1
	print(distance(adj, cost, s, t))
	'''
	#n, p = 25, 0.1
	#adj, w = gen_ER_graph(n, p)
	#s, t = randint(0, n-1), randint(0, n-1)
	#print(s, t)
	#print(distance2(adj, w, s, t))
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
	distance2(adj, cost, s, t, x, y)