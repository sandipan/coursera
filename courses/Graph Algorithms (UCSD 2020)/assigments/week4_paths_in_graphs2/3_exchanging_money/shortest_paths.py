#Uses python3

import sys
import queue

from graphviz import Digraph
id = 0
from random import random, randint

INF = 10**9

def gen_ER_graph2(n, p, c):
	ncs = [randint(2, n) for _ in range(c)]
	cost = {}
	print(ncs)
	adj = [[] for _ in range(sum(ncs))]
	crs = [None for _ in range(c)]
	x = 0
	for i in range(c):
		nc = ncs[i]
		for u in range(x, x + nc):
			for v in range(u+1, x + nc):
				if random() < p:
					w = randint(-45,45)
					if random() < 1: #0.5:
						adj[u].append(v)
						cost[u,v] = w
					else:
						adj[v].append(u)
						cost[v,u] = w
		crs[i] = randint(x, x+nc-1)
		x += nc
	
	#for i in range(c-1):
	#	if random() < 0.5:
	#		adj[crs[i]].append(crs[i+1])
	#	else:
	#		adj[crs[i+1]].append(crs[i])
	
	print(adj)
	return adj, cost
	
def gen_ER_graph(n, p):
	adj = [[] for _ in range(n)]
	cost = {}
	for u in range(n):
		for v in range(u+1, n):
			if random() < p:
				w = randint(-45,45)
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

def draw_graph(adj, cost, colors, d, x, iter=None, cyc=False):
	global id
	s = Digraph('graph', format='jpg')
	#s.attr(compound='true')
	edges = set([])
	for u in range(len(adj)):
		s.node(str(u), str(u) + '\n(' + str(d[u] if d[u] < INF else '∞') + ')', style='filled', color=colors[u])
	for u in range(len(adj)):
		for v in adj[u]:
			if not (u, v) in edges: 
				edges.add((u, v))
				s.edge(str(u), str(v), color=colors.get((u,v), 'lightgray'), label=str(cost.get((u,v), '∞')))
	s.attr(label='\n\nDetecting Negative Cycles with Shortest Distance from {} to with Bellman-Ford'.format(x) + (', iter = ' + str(iter) if iter else '') + '\nNegative Cycle found = {}'.format(cyc))
	s.attr(fontsize='15')
	s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1
	
def shortet_paths2(adj, cost, s, distance, reachable, shortest):
	#write your code here
	inf = float('inf')  #10**19
	n = len(adj)
	d = [inf]*n
	parent = [None]*n
	d[s] = 0
	colors = {v:'floralwhite' for v in range(len(adj))}
	colors[s] = 'pink'
	draw_graph(adj, cost, colors, d, s)
	for k in range(n-1):
		changed = False
		for u in range(n):
			changed2 = False
			for v in adj[u]:
				if d[v] > d[u] + cost[u,v]:
					d[v] = d[u] + cost[u,v]
					colors[u, v] = colors[v, u] = 'dimgrey'
					parent[v] = u
					changed = changed2 = True
			if changed2:
				draw_graph(adj, cost, colors, d, s, k+1)
		if not changed:
			break
	cvs = []
	neg_cycle = False
	for u in range(n):
		changed2 = False
		for v in adj[u]:
			if d[v] > d[u] + cost[u,v]:
				d[v] = d[u] + cost[u,v]
				colors[u, v] = colors[v, u] = 'dimgrey'
				cvs.append(v)
				parent[v] = u
				print('here', u, v)
				neg_cycle = True
				changed2 = True
		if changed2:
			draw_graph(adj, cost, colors, d, s, n)
				#break
		#if cv >= 0:
		#	break
	print(neg_cycle)
	
	visited = [False]*n
	for cv in cvs:
		q = [cv]
		while len(q) > 0:
			u = q.pop(0)
			visited[u] = True
			shortest[u] = 0
			for v in adj[u]:
				if not visited[v]:
					q.append(v)
	
	
	'''
	print(cvs)
	for cv in cvs:
		neg_cycle = []
		start = cv
		while True:
			neg_cycle.append(cv)
			#shortest[cv] = 0
			cv = parent[cv]
			if cv == start:
				break
		print(neg_cycle)
		for i in range(len(neg_cycle)-1):
			u, v = neg_cycle[i], neg_cycle[i+1]
			colors[v] = 'red'
			colors[u,v] = colors[v,u] = 'red'
		draw_graph(adj, cost, colors, d, s)
	
	'''
	
	for u in range(n):
		if d[u] < inf:
			reachable[u] = 1
			if shortest[u]:
				distance[u] = d[u]
				
	for x in range(n):
		if reachable[x] == 0:
			colors[x] = 'dimgrey'
		elif shortest[x] == 0:
			colors[x] = 'red'
		else:
			colors[x] = 'green'

	draw_graph(adj, cost, colors, d, s, cyc=neg_cycle)
	
def shortet_paths(adj, cost, s, distance, reachable, shortest):
	#write your code here
	inf = float('inf')  #10**19
	n = len(adj)
	d = [inf]*n
	parent = [None]*n
	d[s] = 0
	for k in range(n-1):
		changed = False
		for u in range(n):
			for i in range(len(adj[u])):
				v = adj[u][i]
				if d[v] > d[u] + cost[u][i]:
					d[v] = d[u] + cost[u][i]
					changed = True
					#parent[v] = u
		if not changed:
			break
	cvs = []
	for u in range(n):
		for i in range(len(adj[u])):
			v = adj[u][i]
			if d[v] > d[u] + cost[u][i]:
				d[v] = d[u] + cost[u][i]
				cvs.append(v)
				#break
		#if cv >= 0:
		#	break
	
	visited = [False]*n
	for cv in cvs:
		q = [cv]
		while len(q) > 0:
			u = q.pop(0)
			visited[u] = True
			shortest[u] = 0
			for v in adj[u]:
				if not visited[v]:
					q.append(v)
	
	'''
	for cv in cvs:
		neg_cycle = []
		start = cv
		while True:
			neg_cycle.append(cv)
			shortest[cv] = 0
			cv = parent[cv]
			if cv == start:
				break
		#print(neg_cycle)
	'''
	
	for u in range(n):
		if d[u] < inf:
			reachable[u] = 1
			if shortest[u]:
				distance[u] = d[u]

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
    s = data[0]
    s -= 1
    distance = [10**19] * n
    reachable = [0] * n
    shortest = [1] * n
    shortet_paths(adj, cost, s, distance, reachable, shortest)
    for x in range(n):
        if reachable[x] == 0:
            print('*')
        elif shortest[x] == 0:
            print('-')
        else:
            print(distance[x])
    '''
    n, p = 15, 0.175
    distance = [10**19] * n
    reachable = [0] * n
    shortest = [1] * n
    adj, w = gen_ER_graph(n, p)
    #n, p, c = 10, 0.75, 2
    #adj, w = gen_ER_graph2(n, p, c)
    s = randint(0, n-1)
    print(s)
    print(shortet_paths2(adj, w, s, distance, reachable, shortest))
