#!/usr/bin/python3

import sys
import queue
import math

class AStar2:
	def __init__(self, n, adj, cost, x, y):
		# See the explanations of these fields in the starter for friend_suggestion        
		self.n = n;
		self.adj = adj
		self.cost = cost
		self.inf = n*10**6
		self.d = [self.inf]*n
		self.p = {}
		self.visited = [False]*n
		self.workset = []
		# Coordinates of the nodes
		self.x = x
		self.y = y

	def clear(self):
		for v in self.workset:
			self.d[v] = self.inf
			self.visited[v] = False;
		del self.workset[0:len(self.workset)]

	def visit(self, q, v, s, t, dist, measure):
		# Implement this method yourself
		self.d[v] = dist
		q.put((self.d[v] + measure(v, t) - measure(s, t), v))
		#q.put((self.d[v] + measure(v, t), v))
		self.workset.append(v)
		
	def potential(self, v, t):
		if v not in self.p:
			self.p[v] = math.sqrt((self.x[v]-self.x[t])**2+(self.y[v]-self.y[t])**2)
		return self.p[v]

	def process(self, q, u, t):
		for v, w in zip(self.adj[u], self.cost[u]):
			if not self.visited[v]:
				if self.d[v] > self.d[u] + w:
					self.visit(q, v, s, t, 	self.d[u] + w, self.potential)

	# Returns the distance from s to t in the graph
	def query(self, s, t):
		self.clear()
		q = queue.PriorityQueue()
		# Implement the rest of the algorithm yourself		
		n = len(self.adj)
		d = [self.inf]*n
		self.visit(q, s, s, t, 0, self.potential)
		while not q.empty():
			u = q.get()[1]
			if self.visited[u]: continue
			#print(u, self.p)
			if u == t:
				return (self.d[t] if self.d[t] != self.inf else -1)
			self.process(q, u, t)
			self.visited[u] = True
			#self.workset.remove(u)
		
		return -1
		
class AStar:
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
        for u, w in zip(adj[v], cost[v]):
            if not self.visited[u]:
                self.visit(q, u, self.d[v] + w, self.potential(u, t))
            
    def query(self, s, t):
        self.clear()
        q = queue.PriorityQueue()
        self.visit(q, s, 0, self.potential(s, t))
        while not q.empty():
            v = self.extract_min(q)
            if v == t:
                return (self.d[t] if self.d[t] != self.inf else -1)
            if not self.visited[v]:
                self.process(q, v, t)
                self.visited[v] = True
        return -1

def readl():
    return map(int, sys.stdin.readline().split())

if __name__ == '__main__':
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
