# python3
from graphviz import Digraph
id = 0

class MaxMatching2:
	def read_data(self):
		n, m = map(int, input().split())
		adj_matrix = [list(map(int, input().split())) for i in range(n)]
		return adj_matrix

	def write_response(self, matching):
		line = [str(-1 if x == -1 else x + 1) for x in matching]
		print(' '.join(line))

	def find_matching(self, adj_matrix):
		# Replace this code with an algorithm that finds the maximum
		# matching correctly in all cases.

		class Edge:

			def __init__(self, u, v, capacity):
				self.u = u
				self.v = v
				self.capacity = capacity
				self.flow = 0

		# This class implements a bit unusual scheme for storing edges of the graph,
		# in order to retrieve the backward edge for a given edge quickly.
		class FlowGraph:

			def __init__(self, n):
				# List of all - forward and backward - edges
				self.edges = []
				# These adjacency lists store only indices of edges in the edges list
				self.graph = [[] for _ in range(n)]

			def add_edge(self, from_, to, capacity):
				# Note that we first append a forward edge and then a backward edge,
				# so all forward edges are stored at even indices (starting from 0),
				# whereas backward edges are stored at odd indices.
				forward_edge = Edge(from_, to, capacity)
				backward_edge = Edge(to, from_, 0)
				self.graph[from_].append(len(self.edges))
				self.edges.append(forward_edge)
				self.graph[to].append(len(self.edges))
				self.edges.append(backward_edge)

			def size(self):
				return len(self.graph)

			def get_ids(self, from_):
				return self.graph[from_]

			def get_edge(self, id):
				return self.edges[id]

			def add_flow(self, id, flow):
				# To get a backward edge for a true forward edge (i.e id is even), we should get id + 1
				# due to the described above scheme. On the other hand, when we have to get a "backward"
				# edge for a backward edge (i.e. get a forward edge for backward - id is odd), id - 1
				# should be taken.
				#
				# It turns out that id ^ 1 works for both cases. Think this through!
				self.edges[id].flow += flow
				self.edges[id ^ 1].flow -= flow
				
		def max_flow(graph, from_, to, n1):
			flow = 0
			n = len(graph.graph)
			colors = {}
			#colors = {v:'floralwhite' for v in range(n)}
			#colors[from_] = colors[to] = 'red'
			draw_graph(graph, from_, to, n1, colors, flow, [], None)
			flow_edges = []
			# your code goes here
			while True:
				inf = float('Inf')
				n = len(graph.graph)
				d = [inf]*n
				queue = [from_]
				d[from_] = 0
				par = [-1]*n
				for e in graph.edges:
					colors[e.u, e.v] = 'lightgrey'
				while len(queue) > 0:
					u = queue.pop(0)
					for eid in graph.graph[u]:
						e = graph.get_edge(eid)
						u, v, c, f = e.u, e.v, e.capacity, e.flow
						if d[v] ==  inf and f < c:
							queue.append(v)
							d[v] = d[u] + 1
							par[v] = (u, eid, c - f)
							#colors[v] = 'lightgrey'
							colors[u, v] = 'dimgrey'	#colors[v, u] = 
							#colors[from_] = colors[to] = 'red'
							draw_graph(graph, from_, to, n1, colors, flow, [], None)							
							if v == to:
								break
				if par[to] != -1:
					min_flow = inf
					v = to
					path = []
					while True:
						u, eid, f = par[v]
						min_flow = min(min_flow, f)
						path = [eid] + path
						if u == from_:
							break
						v = u
				else:
					break
				
				flow_path = []
				for eid in path:
					graph.add_flow(eid, min_flow)
					e = graph.get_edge(eid)
					flow_edges.append((e.u, e.v, min_flow))
					flow_path.append((e.u, e.v))
					
				flow += min_flow
				#colors[from_] = colors[to] = 'red'
				draw_graph(graph, from_, to, n1, colors, flow, flow_path, min_flow)
			
			for e in graph.edges:
				colors[e.u, e.v] = 'lightgrey'
			for u, v, _ in flow_edges:
				colors[u,v] = 'red'
				#print(u,v)
			draw_graph(graph, from_, to, n1, colors, flow, [], None, True)
	
			return flow, flow_edges

		def draw_graph(graph, from_, to, n1, colors, flow, flow_path, min_flow, found=False):
			global id
			s = Digraph('graph', format='jpg')
			s.attr(rankdir='LR') #, size='8,5')
			n = len(graph.graph)
			s.node(str(from_), color='red', shape='doublecircle')
			with s.subgraph() as ss:
				for i in range(n1):
					ss.attr(rank='same')
					ss.node(str(i), style='filled', color='lightblue2')
			with s.subgraph() as ss:
				for i in range(n1, n-2):
					ss.attr(rank='same')
					ss.node(str(i), style='filled', color='palegreen1')
			s.node(str(to), color='red', shape='doublecircle')
			#s.attr(compound='true')
			for e in graph.edges:
				#if e.flow >= 0 and e.capacity > 0:
				if min_flow != None:
					s.edge(str(e.u), str(e.v), label='({}/{})'.format(e.flow, e.capacity), color='red' if (e.u, e.v) in flow_path else 'lightgray')
				else:
					s.edge(str(e.u), str(e.v), label='({}/{})'.format(e.flow, e.capacity), color=colors.get((e.u, e.v), 'lightgray'))		
			s.attr(label='\n\Bipartite Matching with MaxFlow (Edmond-Karp Algorithm)\n' + \
					('Bottleneck flow in the Augmenting path (Residual graph) = {}\n'.format(min_flow) if min_flow != None else 'Finding an Augmenting path in the Residual graph using BFS\n') + \
					('Max Matching / Max Flow = {}' if found else '#Matches / Flow so far = {}').format(flow))
			s.attr(fontsize='25')
			s.render('out/graph_{:03d}'.format(id), view=False)
			id += 1
			
			
		n = len(adj_matrix)
		m = len(adj_matrix[0])
		matching = [-1] * n
        
		graph = FlowGraph(n + m + 2)
		for i in range(n):
			for j in range(m):
				if adj_matrix[i][j]:
					graph.add_edge(i, n+j, 1)
		for i in range(n):
			graph.add_edge(n+m, i, 1)
			#graph.add_edge(-1, i, 1)
		for j in range(m):
			graph.add_edge(n+j, n+m+1, 1)
			#graph.add_edge(n+j, n+m, 1)
			
		flow, flow_edges = max_flow(graph, n+m, n+m+1, n)
		#flow, flow_edges = max_flow(graph, -1, n+m, n)
		
		colors = {}
		for e in graph.edges:
			colors[e.u,e.v]	= 'lightgray'
			
		for (u, v, f) in flow_edges:
			if u < n:
				matching[u] = v - n # a node may have multiple matchings
				#print(u, v, v-n)
		for u in range(n):
			colors[u, n+matching[u]] = 'red'
				
		draw_graph(graph, n+m, n+m+1, n, colors, flow, [], None, True)
		
		return matching
	
		

	def solve(self):
		adj_matrix = self.read_data()
		matching = self.find_matching(adj_matrix)
		self.write_response(matching)
		
	def solve2(self):
		lines = open('tests/11').read().splitlines()
		n, m = map(int, lines[0].split())
		adj_matrix = [list(map(int, lines[i].split())) for i in range(1, n+1)]
		matching = self.find_matching(adj_matrix)
		self.write_response(matching)


class MaxMatching:
	def read_data(self):
		n, m = map(int, input().split())
		adj_matrix = [list(map(int, input().split())) for i in range(n)]
		return adj_matrix

	def write_response(self, matching):
		line = [str(-1 if x == -1 else x + 1) for x in matching]
		print(' '.join(line))

	def find_matching(self, adj_matrix):
		# Replace this code with an algorithm that finds the maximum
		# matching correctly in all cases.

		class Edge:

			def __init__(self, u, v, capacity):
				self.u = u
				self.v = v
				self.capacity = capacity
				self.flow = 0

		# This class implements a bit unusual scheme for storing edges of the graph,
		# in order to retrieve the backward edge for a given edge quickly.
		class FlowGraph:

			def __init__(self, n):
				# List of all - forward and backward - edges
				self.edges = []
				# These adjacency lists store only indices of edges in the edges list
				self.graph = [[] for _ in range(n)]

			def add_edge(self, from_, to, capacity):
				# Note that we first append a forward edge and then a backward edge,
				# so all forward edges are stored at even indices (starting from 0),
				# whereas backward edges are stored at odd indices.
				forward_edge = Edge(from_, to, capacity)
				backward_edge = Edge(to, from_, 0)
				self.graph[from_].append(len(self.edges))
				self.edges.append(forward_edge)
				self.graph[to].append(len(self.edges))
				self.edges.append(backward_edge)

			def size(self):
				return len(self.graph)

			def get_ids(self, from_):
				return self.graph[from_]

			def get_edge(self, id):
				return self.edges[id]

			def add_flow(self, id, flow):
				# To get a backward edge for a true forward edge (i.e id is even), we should get id + 1
				# due to the described above scheme. On the other hand, when we have to get a "backward"
				# edge for a backward edge (i.e. get a forward edge for backward - id is odd), id - 1
				# should be taken.
				#
				# It turns out that id ^ 1 works for both cases. Think this through!
				self.edges[id].flow += flow
				self.edges[id ^ 1].flow -= flow
				
		def max_flow(graph, from_, to):
			flow = 0
			flow_edges = []
			# your code goes here
			while True:
				inf = float('Inf')
				n = len(graph.graph)
				d = [inf]*n
				queue = [from_]
				d[from_] = 0
				par = [-1]*n
				while len(queue) > 0:
					u = queue.pop(0)
					for eid in graph.graph[u]:
						e = graph.get_edge(eid)
						u, v, c, f = e.u, e.v, e.capacity, e.flow
						if d[v] ==  inf and f < c:
							queue.append(v)
							d[v] = d[u] + 1
							par[v] = (u, eid, c - f)
							if v == to:
								break
				if par[to] != -1:
					min_flow = inf
					v = to
					path = []
					while True:
						u, eid, f = par[v]
						min_flow = min(min_flow, f)
						path = [eid] + path
						if u == from_:
							break
						v = u
				else:
					break
						
				for eid in path:
					graph.add_flow(eid, min_flow)
					e = graph.get_edge(eid)
					flow_edges.append((e.u, e.v, min_flow))
					
				flow += min_flow
					
			return flow, flow_edges
			
		n = len(adj_matrix)
		m = len(adj_matrix[0])
		matching = [-1] * n
        
		graph = FlowGraph(n + m + 2)
		for i in range(n):
			for j in range(m):
				if adj_matrix[i][j]:
					graph.add_edge(i, n+j, 1)
		for i in range(n):
			graph.add_edge(n+m, i, 1)
		for j in range(m):
			graph.add_edge(n+j, n+m+1, 1)
			
		flow, flow_edges = max_flow(graph, n+m, n+m+1)
		
		for (u, v, f) in flow_edges:
			if u < n:
				matching[u] = v - n
		
		return matching

	def solve(self):
		adj_matrix = self.read_data()
		matching = self.find_matching(adj_matrix)
		self.write_response(matching)

def run_tests():
	from glob import glob
	files = glob('tests/*')
	files_answer = sorted(list(filter(lambda x: x.endswith('.a'), files)))
	files_input = sorted(list(set(files) - set(files_answer)))
	#print(len(files_input), len(files_answer))
	for i in range(len(files_input)):
		lines = open(files_input[i]).read().splitlines()
		n, m = map(int, lines[0].split())
		adj_matrix = [list(map(int, lines[i].split())) for i in range(1, n+1)]
		max_matching = MaxMatching()
		matching = max_matching.find_matching(adj_matrix)
		matching = [-1 if x == -1 else x + 1 for x in matching]
		res = list(map(int, open(files_answer[i]).read().split()))
		passed = (matching == res)
		try:
			print(i, n, m, 'passed' if passed else 'failed') #matching, res, 
			#if not passed:
				#print(adj_matrix)
			#	break
		except Exception as e:
			print(str(e))

import numpy as np			
def gen_random_matching(n, m, dmax=5):
	edges = []
	for u in range(n):
		d = np.random.randint(1, dmax)
		for v in np.random.choice(range(m), d):
			edges.append((u, v))
	adj = np.zeros((n,m), dtype=int)		
	print(n, m)
	for u, v in edges:
		adj[u, v] = 1
	print(adj)
	
if __name__ == '__main__':
	#max_matching = MaxMatching()
	max_matching = MaxMatching2()
	max_matching.solve()
	#run_tests()
	#max_matching = MaxMatching2()
	#max_matching.solve2()
	#gen_random_matching(5, 6)
	