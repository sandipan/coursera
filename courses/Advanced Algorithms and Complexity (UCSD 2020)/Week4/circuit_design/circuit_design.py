# python3
import sys
from graphviz import Digraph
id = 0
import numpy as np

import threading
import collections, itertools
try:
    from collections.abc import MutableSet  # noqa
except ImportError:
    from collections import MutableSet  # noqa
	
#import resource

#resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))

sys.setrecursionlimit(10**6)
threading.stack_size(2**26)

n, m = map(int, input().split())
clauses = [ list(map(int, input().split())) for i in range(m) ]

# This solution tries all possible 2^n variable assignments.
# It is too slow to pass the problem.
# Implement a more efficient algorithm here.
'''
def isSatisfiable():
    for mask in range(1<<n):
        result = [ (mask >> i) & 1 for i in range(n) ]
        formulaIsSatisfied = True
        for clause in clauses:
            clauseIsSatisfied = False
            if result[abs(clause[0]) - 1] == (clause[0] < 0):
                clauseIsSatisfied = True
            if result[abs(clause[1]) - 1] == (clause[1] < 0):
                clauseIsSatisfied = True
            if not clauseIsSatisfied:
                formulaIsSatisfied = False
                break
        if formulaIsSatisfied:
            return result
    return None
'''
clock = 1

def isSatisfiable():

	def neg(i):
		return -i if i < 0 else n + abs(i)
		
	def pos(i):
		return i if i > 0 else n + abs(i)
		
	def create_implication_graph():
		f  = ''
		adj = [[] for _ in range(2*n+1)]
		for [x, y] in clauses:
			adj[neg(x)].append(pos(y)) # ~x -> y
			adj[neg(y)].append(pos(x)) # ~y -> x
			#f += '({} OR {}) AND '.format(('x' if x > 0 else '¬x') + str(abs(x)), ('x' if y > 0 else '¬x') + str(abs(y)))
			f += '({},{}) '.format(('x' if x > 0 else '¬x') + str(abs(x)), ('x' if y > 0 else '¬x') + str(abs(y)))
		return adj, f
		
	def find_assignments(adj):
	
		#write your code here
		
		def reverse_graph(adj):
			n = len(adj)
			new_adj = [ [] for _ in range(n)]
			for i in range(n):
				for j in adj[i]:
					if not i in new_adj[j]:
						new_adj[j].append(i)
			return new_adj
		
		def dfs_visit(adj, u, cc=None):
			global clock
			visited[u] = True
			if cc != None: cc.append(u)
			previsit[u] = clock
			clock += 1
			for v in adj[u]:
				if not visited[v]:
					dfs_visit(adj, v, cc)
			postvisit[u] = clock
			clock += 1
		
		n = len(adj)
		colors = {u:'floralwhite' for u in range(n)}
		visited = [False]*n
		previsit = [0]*n
		postvisit = [0]*n
		for v in range(n):
			if not visited[v]:
				dfs_visit(adj, v)
		post_v = [x for _, x in sorted(zip(postvisit, range(n)), key=lambda pair: pair[0], reverse=True)]
		#for v in post_v:
		#	print(v, previsit[v], postvisit[v])
		#print(post_v)
		nv = n//2
		result = [None] * nv
		rev_adj = reverse_graph(adj)
		visited = [False]*(n)
		ccolors = ['aquamarine', 'burlywood', 'cadetblue1', 'chartreuse', 'darkgoldenrod1', 'darkolivegreen1', 'cyan3', 'gold1', 'steelblue', 'deemgray', 'brown1']
		i = 0
		for v in post_v:
			cc = []
			if not visited[v]:
				dfs_visit(rev_adj, v, cc)
				print('here', v, cc)
				for x in cc:
					colors[x] = ccolors[i]
				i += 1
				draw_graph(adj, f, colors)
		visited = [False]*(n)
		colors = {u:'floralwhite' for u in range(n)}
		for v in post_v:
			cc = []
			if not visited[v]:
				dfs_visit(rev_adj, v, cc)
			#print('here', v, cc)
			for x in range(1, nv+1):
				if x in cc and neg(x) in cc:
					colors[x] = colors[neg(x)] = 'red'
					draw_graph(adj, f, colors, 'Not Satisfiable!')
					return None
			for x in cc[::-1]:
				if x == 0: continue
				#print(x)
				neg_x = False
				if x > nv:
					neg_x, x = True, x - nv
					#print(x, neg_x)
				if result[x-1] == None:
					result[x-1] = 0 if neg_x else 1
					colors[x] = 'green' if neg_x else 'red'
					colors[nv + x] = 'green' if colors[x] == 'red' else 'red'
					draw_graph(adj, f, colors, 'A Satisfying Assignment: ' + ' '.join(['' if result[x-1] == None else '{}={}'.format(node2id(x), 'F' if result[x-1] else 'T') for x in range(1, nv+1)]))
				#print(result)
		return result
			
	adj, f = create_implication_graph()
	colors = {u:'floralwhite' for u in range(len(adj))}
	print(adj)
	draw_graph(adj, f, colors)
	return find_assignments(adj)

def node2id(x):
	return 'x' + str(x) if x <= n else '¬x' + str(x - n)
	
def draw_graph(adj, f, colors, res=''):
	global id
	s = Digraph('graph', format='jpg')
	edges = set([])
	for u in range(1, len(adj)):
		s.node(node2id(u), style='filled', color=colors[u])
	for u in range(1, len(adj)):
		for v in adj[u]:
			if not (u, v) in edges: 
				edges.add((u, v))
				s.edge(node2id(u), node2id(v), color=colors.get((u,v), 'lightgray'))
	s.attr(label=r'\n\nImplication Graph for 2-SAT (finding SCCs / var assignment)\nCNF clauses: {}\n'.format(f) + res) # f[:-4]
	s.attr(fontsize='15')
	s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1


class Ordered_Set(MutableSet):

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            current = end[1]
            current[2] = end[1] = self.map[key] = [key, current, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        current = end[2]
        while current is not end:
            yield current[0]
            current = current[2]

    def __reversed__(self):
        end = self.end
        current = end[1]
        while current is not end:
            yield current[0]
            current = current[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, Ordered_Set):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)


def post_orders_ss(adjacents):

    def dfs(node, order, traversed):
        traversed.add(node)
        for adj in adjacents[node]:
            if adj in traversed:
                continue
            dfs(adj, order, traversed)
        if node in vertices:
            vertices.remove(node)
        order.add(node)

    post_order = Ordered_Set([])
    traversed = set([])
    vertices = set([node for node in range(len(adjacents))])

    while True:
        dfs(vertices.pop(), post_order, traversed)
        if len(post_order) == len(adjacents):
            break

    assert len(post_order) == len(adjacents)

    return list(post_order)

def connected_component(adjacents, node, found):

    connected = set([])
    def dfs(node, connected):
        connected.add(node)
        found.add(node)
        found.add(node)
        for adj in adjacents[node]:
            if adj in found or adj in connected:
                continue
            dfs(adj, connected)

    dfs(node, connected)
    return connected

def analyze_connected_components(n, adjacents, reverse, var_map):

    order = post_orders_ss(reverse)


    order_pointer = len(order) - 1
    found = set([])
    ccs = []
    while order_pointer >= 0:
        if order[order_pointer] in found:
            order_pointer -= 1
            continue

        ccs.append(connected_component(adjacents, order[order_pointer], found))

    assert len(found) == len(adjacents), 'found {0} nodes, but {1} were specified'.format(len(found), n)
    return ccs

def build_implication_graph(n, clauses):

    edges = []
    var_dict =  {}
    node_dict = {}
    node_num = 0
    adjacents = [[] for _ in range(2*n)]
    reversed_adjs = [[] for _ in range(2*n)]

    for clause in clauses:
        #if len(clause) == 1:
        #    assert False, 'should be two terms in the clause'

        left = clause[0]
        right = clause[1]
        for term in [left, right]:
            if not term in node_dict:
                var_dict[node_num] = term
                node_dict[term] = node_num
                node_num += 1
            if not -term in node_dict:
                var_dict[node_num] = -term
                node_dict[-term] = node_num
                node_num += 1

        adjacents[node_dict[-left]].append(node_dict[right])
        reversed_adjs[node_dict[right]].append(node_dict[-left])

        adjacents[node_dict[-right]].append(node_dict[left])
        reversed_adjs[node_dict[left]].append(node_dict[-right])


    return edges, adjacents[:node_num], reversed_adjs[:node_num], node_dict, var_dict

def is_satisfiable(n, m, clauses):
    edges, implication_g, reversed_imp_g, node_map, var_map = build_implication_graph(n, clauses)

    ccs = analyze_connected_components(n, implication_g, reversed_imp_g, var_map)

    #print(ccs)

    result = collections.defaultdict(lambda: None)
    for cc in ccs:
        cc_vars = set([])
        for node in cc:
            lit = var_map[node]
            if abs(lit) in cc_vars:
                return None
            else:
                cc_vars.add(abs(lit))

            if result[abs(lit)] is None:
                if lit < 0:
                    result[abs(lit)] = 0
                else:
                    result[abs(lit)] = 1

    return result

def isSatisfiable2():
	return is_satisfiable(n, m, clauses)
	
def gen_clause(n, m):
	print(n, m)
	for i in range(m):
		x, y = np.random.choice(range(1, n+1), 2)
		if np.random.rand() < 0.5:
			x *= -1
		if np.random.rand() < 0.5:
			y *= -1
		print (x, y)

#gen_clause(4, 12)

result = isSatisfiable()
if result is None:
    print("UNSATISFIABLE")
else:
    print("SATISFIABLE");
    print(" ".join(str(-i-1 if result[i] else i+1) for i in range(n)))