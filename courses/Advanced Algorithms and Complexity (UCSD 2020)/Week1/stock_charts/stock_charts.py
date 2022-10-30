# python3
class StockCharts:
	def read_data(self):
		n, k = map(int, input().split())
		stock_data = [list(map(int, input().split())) for i in range(n)]
		return stock_data

	def write_response(self, result):
		print(result)

	'''
	def min_charts(self, stock_data):
		# Replace this incorrect greedy algorithm with an
		# algorithm that correctly finds the minimum number
		# of charts on which we can put all the stock data
		# without intersections of graphs on one chart.
		n = len(stock_data)
		k = len(stock_data[0])
		charts = []
		for new_stock in stock_data:
			added = False
			for chart in charts:
				fits = True
				for stock in chart:
					above = all([x > y for x, y in zip(new_stock, stock)])
					below = all([x < y for x, y in zip(new_stock, stock)])
					if (not above) and (not below):
						fits = False
						break
				if fits:
					added = True
					chart.append(new_stock)
					break
			if not added:
				charts.append([new_stock])
		return len(charts)
	'''
	
	def min_charts(self, stock_data):
	
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
					
				flow += min_flow
					
			return flow
			
		n = len(stock_data)
		k = len(stock_data[0])
		# create DAG
		adj = [[] for _ in range(n)]
		for i in range(n):
			for j in range(n):
				if i == j: continue
				above = all([stock_data[i][p] > stock_data[j][p] for p in range(k)])
				if above:
					adj[i].append(j)
					
		# print(adj)
		# create bipartite
		
		graph = FlowGraph(2*n + 2)
		for i in range(n):
			for j in adj[i]:
				graph.add_edge(i, n+j, 1)
		for i in range(n):
			graph.add_edge(2*n, i, 1)
			graph.add_edge(n+i, 2*n+1, 1)
			
		flow = max_flow(graph, 2*n, 2*n+1)
		
		return n - flow

	def solve(self):
		stock_data = self.read_data()
		result = self.min_charts(stock_data)
		self.write_response(result)

def run_tests():
	from glob import glob
	files = glob('tests/*')
	files_answer = sorted(list(filter(lambda x: x.endswith('.a'), files)))
	files_input = sorted(list(set(files) - set(files_answer)))
	#print(len(files_input), len(files_answer))
	for i in range(len(files_input)):
		lines = open(files_input[i]).read().splitlines()
		n, k = map(int, lines[0].split())
		stock_data = [list(map(int, lines[i].split())) for i in range(1, n+1)]
		stock_charts = StockCharts()
		result = stock_charts.min_charts(stock_data)
		res = int(open(files_answer[i]).read())
		passed = (result == res)
		try:
			print(i, n, k, result, res, 'passed' if passed else 'failed') #matching, res, 
			#if not passed:
				#print(adj_matrix)
			#	break
		except Exception as e:
			print(str(e))

if __name__ == '__main__':
	stock_charts = StockCharts()
	stock_charts.solve()
	#run_tests()