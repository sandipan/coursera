#uses python3

import sys
import threading
from graphviz import Graph

# This code is used to avoid stack overflow issues
sys.setrecursionlimit(10**6) # max depth of recursion
threading.stack_size(2**26)  # new thread will get stack of such size

id = 0

class Vertex:
    def __init__(self, weight):
        self.weight = weight
        self.children = []

def post_order_traverse(tree, root, lst):
	if tree[root]:
		for child in tree[root].children:
			post_order_traverse(tree, child, lst)
	lst += [root]  	
	return lst
	
def read_tree():
	size = int(input())
	parent = {}
	tree = [Vertex(w) for w in map(int, input().split())]
	for i in range(1, size):
		a, b = list(map(int, input().split()))
		tree[b-1].children.append(a - 1)
		parent[a-1] = b - 1
	root = list(filter(lambda x: parent.get(x, None) == None, list(range(size))))[0]
	return tree, root

def compute_MIS(tree, root):
	n = len(tree)
	lst = post_order_traverse(tree, root, [])
	M = [-1] * n
	MIS = {}
	for v in lst:
		m1 = sum([0] + [M[u] for u in tree[v].children])
		me1 = sum([MIS[u] for u in tree[v].children], [])
		m2 = tree[v].weight + sum([0]+[M[x] for u in tree[v].children for x in tree[u].children])
		me2 = [v] + sum([MIS[x] for u in tree[v].children for x in tree[u].children], [])
		M[v] = max(m1, m2)
		MIS[v] = me1 if max(m1, m2) == m1 else me2
		draw_graph(tree, M, MIS, v)
	return M, MIS
	
def get_queue(q):
	html = "'''<<table>"
	n =  len(q) if q else 0
	html += '<tr>'
	for i in range(n):
		html += '<td>'
		html += str(q[i])
		html += '</td>'
	html += '</tr>'
	html += "</table>>'''"
	return html

def draw_graph(tree, M, MIS, v):
	global id
	s = Graph('graph_mis', format='jpg')
	#s.attr(compound='true')
	with s.subgraph(name='graph') as c:
		for u in range(len(tree)):
			c.node(str(u), str(u) + '\n(w={})'.format(tree[u].weight), style='filled', color='pink' if u in MIS[v] else 'lightgray', shape='doublecircle' if u == v else 'circle')
		for u in range(len(tree)):
			for x in tree[u].children:
				c.edge(str(u), str(x), color='lightgray')
		c.attr(label='Graph')
		c.attr(fontsize='15')
	with s.subgraph(name='M') as c:
		c.node(str(len(tree)), eval(get_queue(M))) #, style='filled', color='mistyrose')
		c.attr(label='M')
		c.attr(fontsize='15')
	s.attr(label=r'\n\nMaximum Independent Set\ncomputed with Dynamic Programming\nfor the tree rooted at {} = {}, total weight = {}'.format(v, MIS[v], M[v]))
	s.attr(fontsize='15')
	s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1

def main():
	tree, root = read_tree()
	M, MIS = compute_MIS(tree, root)
	print(M)
	print(MIS)
	#weight = MaxWeightIndependentTreeSubset(tree);
	#print(weight)


# This is to avoid stack overflow issues
threading.Thread(target=main).start()
