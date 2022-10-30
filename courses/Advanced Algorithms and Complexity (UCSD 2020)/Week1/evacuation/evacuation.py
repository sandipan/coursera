# python3
from graphviz import Digraph
id = 0

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


def read_data():
    vertex_count, edge_count = map(int, input().split())
    graph = FlowGraph(vertex_count)
    for _ in range(edge_count):
        u, v, capacity = map(int, input().split())
        graph.add_edge(u - 1, v - 1, capacity)
    return graph


def max_flow(graph, from_, to):
	flow = 0
	# your code goes here
	n = len(graph.graph)
	colors = {v:'floralwhite' for v in range(n)}
	colors[from_] = colors[to] = 'red'
	draw_graph(graph, from_, to, colors, flow, [], None)
	flow_edges = []
	while True:
	
		inf = float('Inf')
		d = [inf]*n
		colors = {v:'floralwhite' for v in range(n)}
		colors[from_] = colors[to] = 'red'
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
					colors[v] = 'lightgrey'
					colors[u, v] = 'dimgrey'	#colors[v, u] = 
					colors[from_] = colors[to] = 'red'
					draw_graph(graph, from_, to, colors, flow, [], None)
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
			flow_path.append((e.u, e.v))
			flow_edges.append((e.u, e.v))
			
		flow += min_flow
		colors[from_] = colors[to] = 'red'
		draw_graph(graph, from_, to, colors, flow, flow_path, min_flow)
	
	for e in graph.edges:
		colors[e.u, e.v] = 'lightgrey'
	for u, v in flow_edges:
		colors[u,v] = 'red'
		print(u,v)
	draw_graph(graph, from_, to, colors, flow, [], None, True)
		
	return flow
	
def draw_graph(graph, from_, to, colors, flow, flow_path, min_flow, found=False):
	global id
	s = Digraph('graph', format='jpg')
	s.attr(rankdir='LR', size='8,5')
	#s.attr(compound='true')
	s.node(str(from_), color='red', shape='doublecircle')
	s.node(str(to), color='red', shape='doublecircle')
	for e in graph.edges:
		#if e.flow >= 0 and e.capacity > 0:
		s.node(str(e.u), style='filled', color=colors[e.u])
		s.node(str(e.v), style='filled', color=colors[e.v])
		if min_flow != None:
			s.edge(str(e.u), str(e.v), label='({}/{})'.format(e.flow, e.capacity), color='red' if (e.u, e.v) in flow_path else 'lightgray')
		else:
			s.edge(str(e.u), str(e.v), label='({}/{})'.format(e.flow, e.capacity), color=colors.get((e.u, e.v), 'lightgray'))		
	s.attr(label='\n\MaxFlow with Edmond-Karp Algorithm\n' + ('Bottleneck flow in the Augmenting path (Residual graph) = {}\n'.format(min_flow) if min_flow != None else 'Finding an Augmenting path in the Residual graph using BFS\n') + \
			('Max Flow = {}' if found else 'Flow so far = {}').format(flow))
	s.attr(fontsize='15')
	s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1

def run_tests():
	from glob import glob
	files = glob('tests/*')
	files_answer = sorted(list(filter(lambda x: x.endswith('.a'), files)))
	files_input = sorted(list(set(files) - set(files_answer)))
	#print(len(files_input), len(files_answer))
	for i in range(len(files_input)):
		lines = open(files_input[i]).read().splitlines()
		vertex_count, edge_count = map(int, lines[0].split())
		graph = FlowGraph(vertex_count)
		j = 1
		for _ in range(edge_count):
			u, v, capacity = map(int, lines[j].split())
			graph.add_edge(u - 1, v - 1, capacity)
			j += 1
		flow = max_flow(graph, 0, graph.size() - 1)
		passed = (flow == int(open(files_answer[i]).read()))
		try:
			print(i, vertex_count, edge_count, flow, 'passed' if passed else 'failed')
			if not passed:
				break
		except Exception as e:
			print(str(e))
			
if __name__ == '__main__':
	graph = read_data()
	print(max_flow(graph, 0, graph.size() - 1))
	#run_tests()
	'''
	lines = open('tests/09').read().splitlines()
	vertex_count, edge_count = map(int, lines[0].split())
	graph = FlowGraph(vertex_count)
	j = 1
	for _ in range(edge_count):
		u, v, capacity = map(int, lines[j].split())
		graph.add_edge(u - 1, v - 1, capacity)
		j += 1
	print(max_flow(graph, 0, graph.size() - 1))
	'''