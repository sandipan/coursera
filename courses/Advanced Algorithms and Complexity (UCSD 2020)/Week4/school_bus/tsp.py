from itertools import combinations, permutations
import numpy as np
from graphviz import Graph, Digraph
from time import time
import matplotlib.pylab as plt
import networkx as nx

id = 0
times = []
timesI = []
sol_types = []
INF = np.inf

def read_data():
    n, m = map(int, input().split())
    graph = [[INF] * n for _ in range(n)]
    for _ in range(m):
        u, v, weight = map(int, input().split())
        u -= 1
        v -= 1
        graph[u][v] = graph[v][u] = weight
    return graph

def print_answer(path_weight, path):
    print(path_weight)
    if path_weight == -1:
        return
    print(' '.join(map(str, path)))

def TSP(G):
	start = time()
	n = len(G)
	C = [[INF for _ in range(n)] for __ in range(1 << n)]
	backptr = [[(-1, -1) for _ in range(n)] for __ in range(1 << n)]
	C[1][0] = 0 # {0} <-> 1
	for size in range(1, n):
		for S in combinations(range(1, n), size):
			S = (0,) + S
			k = sum([1 << i for i in S])
			for i in S:
				if i == 0: continue
				for j in S:
					if j == i: continue
					cur_index = k ^ (1 << i)
					cur = C[cur_index][j] + G[j][i] #C[S−{i}][j]
					if cur < C[k][i]:
						C[k][i], backptr[k][i] = cur, (cur_index, j)
						#draw_graph(C, G, S, i, j)
	all_index = (1 << n) - 1
	tsp_sol, next_index = min([(C[all_index][i] + G[0][i], i) for i in range(n)])

	if tsp_sol >= INF:
		return (-1, [])

	tsp_path = []
	cur_index = all_index
	while cur_index != -1:
		tsp_path.insert(0, next_index) # + 1)
		cur_index, next_index = backptr[cur_index][next_index]
	end = time()
	times.append(end-start)
	print('Time taken={}'.format(end-start))
	#draw_graph(C, G, S, i, j, tsp_sol, tsp_path)
	#draw_graph2(G, tsp_sol, tsp_path, end - start)
	return (tsp_sol, tsp_path, end-start)
	
def get_table(C):
	html = "'''<<table>"
	for j in range(len(C[0])):
		html += '<tr>'
		for i in range(len(C)):
			html += '<td>'
			html += str(C[i][j] if C[i][j] != np.inf else '∞')
			html += '</td>'
		html += '</tr>'
	html += "</table>>'''"
	return html
	
def draw_graph2(G, sol, path, t, solI, pathI, tI):
	global id, times, sol_types
	g = nx.DiGraph()
	for u in range(len(G)):
		for v in range(u+1, len(G)):
			g.add_edge(str(u), str(v), weight=G[u][v])
			g.add_edge(str(v), str(u), weight=G[v][u])
	
	plt.figure(figsize=(20,10))
	plt.subplot(121)
	pos = nx.kamada_kawai_layout(g) #spring_layout(g) fruchterman_reingold_layout
	nx.draw_networkx_nodes(g, pos, node_size=400, node_color='#00b4d9')
	nx.draw_networkx_labels(g, pos)
	path += [0]
	nx.draw_networkx_edges(g, pos, edgelist=[(str(u), str(v)) for u in range(len(G)) for v in range(u+1,len(G))], style='dashed', alpha=0.1, arrows=False)
	nx.draw_networkx_edges(g, pos, edgelist=[(str(path[i]), str(path[i+1])) for i in range(len(path)-1)], edge_color='r', arrows=True, arrowsize=50) #, arrowstyle='fancy')
	nx.draw_networkx_edge_labels(g, pos, edge_labels=nx.get_edge_attributes(g,'weight'))
	plt.title('DP (opt cost = {}, time taken={:.6f}s)'.format(sol, t), size=15)
	plt.subplot(122)
	pos = nx.kamada_kawai_layout(g) #spring_layout(g) fruchterman_reingold_layout
	nx.draw_networkx_nodes(g, pos, node_size=400, node_color='#00b4d9')
	nx.draw_networkx_labels(g, pos)
	nx.draw_networkx_edges(g, pos, edgelist=[(str(u), str(v)) for u in range(len(G)) for v in range(u+1,len(G))], style='dashed', alpha=0.1, arrows=False)
	nx.draw_networkx_edges(g, pos, edgelist=[(str(pathI[i]), str(pathI[i+1])) for i in range(len(pathI)-1)], edge_color='r', arrows=True, arrowsize=50) #, arrowstyle='fancy')
	nx.draw_networkx_edge_labels(g, pos, edge_labels=nx.get_edge_attributes(g,'weight'))
	plt.title('ILP (opt cost = {}, time taken={:.6f}s)'.format(solI, tI), size=15)
	plt.suptitle('TSP optimal path with DP and ILP (#nodes = {})'.format(len(G)), size=20)
	plt.savefig('out/graph_{:03d}.png'.format(id))
	plt.close()
	id += 1
	
def draw_graph3(G, sol, paths, t):
	global id, times
	g = nx.DiGraph()
	for u in range(len(G)):
		for v in range(u+1, len(G)):
			g.add_edge(str(u), str(v), weight=G[u][v])
			g.add_edge(str(v), str(u), weight=G[v][u])
	
	plt.figure(figsize=(20,10))
	pos = nx.kamada_kawai_layout(g) #spring_layout(g) fruchterman_reingold_layout
	nx.draw_networkx_nodes(g, pos, node_size=400, node_color='#00b4d9')
	nx.draw_networkx_labels(g, pos)
	nx.draw_networkx_edges(g, pos, edgelist=[(str(u), str(v)) for u in range(len(G)) for v in range(u+1,len(G))], style='dashed', alpha=0.05, arrows=False)
	for path in paths:
		nx.draw_networkx_edges(g, pos, edgelist=[(str(path[i]), str(path[i+1])) for i in range(len(path)-1)], edge_color='r', arrows=True, arrowsize=50) #, arrowstyle='fancy')
	nx.draw_networkx_edge_labels(g, pos, edge_labels=nx.get_edge_attributes(g,'weight'), alpha=0.1, bbox=dict(alpha=0))
	plt.title('TSP optimal path with ILP (#nodes = {}, opt cost = {}, time taken={:.6f}s)'.format(len(G), sol, t), size=20)
	plt.savefig('out/graph_{:03d}.png'.format(id))
	plt.close()
	id += 1

def draw_graph(C, G, S, i, j, sol=None, path=None):
	global id
	s = Graph('graph_tsp', format='jpg') if not path else Digraph('graph_tsp', format='jpg')
	if path == None:
		#s.attr(compound='true')
		for u in range(len(G)):
			for v in range(u+1, len(G)):
				if u != v:
					s.edge(str(u), str(v), label='{}'.format(G[u][v]), color='lightgray' if sorted([u,v]) != sorted([i,j]) else 'red')
		for v in S:
			s.node(str(v), color='red', shape='doublecircle' if v == i else 'circle')
		#s.edge(str(i), str(j), label='{}'.format(G[i][j]), color='red')
	else:
		path += [0]
		for i in range(len(path)-1):
			s.edge(str(path[i]), str(path[i+1]), label='{}'.format(G[path[i]][path[i+1]]), color='red')
	with s.subgraph(name='T') as c:
		c.node('DPT', eval(get_table(C))) #, style='filled', color='mistyrose')
		c.attr(label='T')
		c.attr(fontsize='15')
	s.attr(label=r'\n\nSolving TSP with Dynamic Programming\nTotal cost: {} Path: {}'.format(sol if sol != None else '', path if path else '')) #list(map(lambda x: x-1, path))
	s.attr(fontsize='15')
	s.render('out/graph_{:03d}'.format(id), view=False)
	id += 1
	
def gen_TSP_data(n):
	print(n, n*(n-1)//2)
	for i in range(1, n+1):
		for j in range(i+1, n+1):
			print(i, j, np.random.randint(10,100))


from itertools import product
from mip import Model, xsum, minimize, BINARY, OptimizationStatus			

def TSP2(G):
	global timesI, sol_types
	start = time()
	# number of nodes and list of vertices
	V1 =  range(len(G))
	n, V = len(G), set(V1)
	print(n, V, G)

	model = Model()

	# binary variables indicating if arc (i,j) is used on the route or not
	x = [[model.add_var(var_type=BINARY) for j in V] for i in V]

	# continuous variable to prevent subtours: each city will have a
	# different sequential id in the planned route except the first one
	y = [model.add_var() for i in V]

	# objective function: minimize the distance
	model.objective = minimize(xsum(G[i][j]*x[i][j] for i in V for j in V))

	# constraint : leave each city only once
	for i in V:
		model += xsum(x[i][j] for j in V - {i}) == 1

	# constraint : enter each city only once
	for i in V:
		model += xsum(x[j][i] for j in V - {i}) == 1

	#model += xsum(x[j][i] + x[i][j] for j in V - {i} for i in V) <= 1

	# subtour elimination
	for (i, j) in product(V - {0}, V - {0}):
		if i != j:
			model += y[i] - (n+1)*x[i][j] >= y[j]-n

	# optimizing
	status = model.optimize()
	sol_type = 'None'
	if status == OptimizationStatus.OPTIMAL:
		sol_type = 'OPTIMAL'
		print('optimal solution cost {} found'.format(model.objective_value))
	elif status == OptimizationStatus.FEASIBLE:
		sol_type = 'FEASIBLE'
		print('sol.cost {} found, best possible: {}'.format(model.objective_value, model.objective_bound))
	elif status == OptimizationStatus.NO_SOLUTION_FOUND:
		print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))
	sol_types.append(sol_type)

	# checking if a solution was found
	if model.num_solutions:
		print('Total distance {}'.format(model.objective_value))
		nsol = 1
		k = 0
		done = False
		cycles = []
		while not done:
			nc = k
			cycle = [nc]
			while True:
				nc = [i for i in V if x[nc][i].x >= 0.99][0]
				cycle.append(nc)
				if nc == k:
					cycles.append(cycle)
					V2 = list(V - set(sum(cycles, [])))
					print(V2, cycle) #, V2[0])
					if len(V2) > 0:
						k = V2[0]
					else:
						done = True
					break
				nsol += 1
		print(cycles)
	end = time()
	timesI.append(end-start)
	print('Time taken = {}'.format(end-start))
	return (model.objective_value, cycles, end-start, sol_type)
	
def swap(s, m, n):
	i, j = min(m, n), max(m, n)
	s1 = s.copy()
	while i < j:
		s1[i], s1[j] = s1[j], s1[i]
		i += 1
		j -= 1
	return s1
	
def cost(G, s):
	l = 0
	for i in range(len(s)-1):
		l += G[s[i]][s[i+1]]
	l += G[s[len(s)-1]][s[0]]	
	return l

def plot_graph(G, path, sol, optsol):
	g = nx.DiGraph()
	for u in range(len(G)):
		for v in range(u+1, len(G)):
			g.add_edge(str(u), str(v), weight=G[u][v])
			g.add_edge(str(v), str(u), weight=G[v][u])
	
	pos = nx.kamada_kawai_layout(g) #spring_layout(g) fruchterman_reingold_layout
	nx.draw_networkx_nodes(g, pos, node_size=400, node_color='#00b4d9')
	nx.draw_networkx_labels(g, pos)
	nx.draw_networkx_edges(g, pos, edgelist=[(str(u), str(v)) for u in range(len(G)) for v in range(u+1,len(G))], style='dashed', alpha=0.05, arrows=False)
	path = path + [path[0]]
	nx.draw_networkx_edges(g, pos, edgelist=[(str(path[i]), str(path[i+1])) for i in range(len(path)-1)], edge_color='r', arrows=True, arrowsize=25) #, arrowstyle='fancy')
	nx.draw_networkx_edge_labels(g, pos, edge_labels=nx.get_edge_attributes(g,'weight'), alpha=0.1, bbox=dict(alpha=0))
	plt.title('TSP path with SA, cost = {} OPT cost = {}'.format(sol, optsol), size=15)
	
def TSP_SA(G, P=None):
	global id
	s = list(range(len(G)))
	c = cost(G, s)
	ntrial = 1
	naccept = 0
	costs = [c]
	paccs = []
	T = 40 #30
	alpha = 0.995
	temps = [T]
	pacc = 0.1
	#optcost, _, _ = TSP2(graph)
	plt.rcParams.update({'font.size': 20})
	bc = np.inf
	while ntrial <= 25000:
		n = np.random.randint(0, len(G))
		while True:
			m = np.random.randint(0, len(G))
			if n != m:
				break
		print(ntrial, T, s)
		s1 = swap(s, m, n)
		c1 = cost(G, s1)
		if c1 < c:
			s, c = s1, c1
		else:
			if np.random.rand() < np.exp(-(c1 - c)/T):
				s, c = s1, c1
				naccept += 1
		costs.append(c)
		paccs.append(naccept / ntrial)
		if c < bc and ntrial % 100 == 0:
			bc = c
			plt.figure(figsize=(20,20))
			if P:
				plt.subplot(211)
				plt.plot([x[0] for x in P], [x[1] for x in P], 'r.', markersize=20)
				for i, p in enumerate(P):
					plt.annotate(str(i), (p[0], p[1]))
				s1 = s + [s[0]]
				print(s)
				plt.plot([P[x][0] for x in s1], [P[x][1] for x in s1], 'g-', label='SA', lw=5, alpha=0.4)
				plt.legend()
				plt.title('TSP with SA cost = {:.03f}, iter = {}'.format(c, ntrial), size=20)
				plt.grid()
				plt.subplot(212), plt.plot(range(ntrial+1), costs), plt.xlabel('#trials', size=15), plt.ylabel('cost', size=15), plt.grid()
			else:
				plt.subplot(221), plot_graph(G, s, c, optcost)
				plt.subplot(222), plt.plot(range(ntrial+1), costs), plt.xlabel('#trials', size=15), plt.ylabel('cost', size=15), plt.grid()
				plt.subplot(223), plt.plot(range(ntrial), paccs), plt.xlabel('#trials', size=15), plt.ylabel('acceptance prob', size=15), plt.grid()
				plt.subplot(224), plt.plot(range(ntrial), temps), plt.xlabel('#trials', size=15), plt.ylabel('temperature', size=15), plt.grid()
			plt.savefig('out/graph_{:03d}.png'.format(id))
			plt.close()
			id += 1			
		T = alpha*T
		temps.append(T)
		ntrial += 1

def do_crossover(s1, s2, m):
	s1 = s1.copy()
	s2 = s2.copy()
	c1 = s2.copy()
	for i in range(m, len(s1)):
		c1.remove(s1[i])
	for i in range(m, len(s1)):
		c1.append(s1[i])
	c2 = s1.copy()
	for i in range(m, len(s2)):
		c2.remove(s2[i])
	for i in range(m, len(s2)):
		c2.append(s2[i])	
	return (c1, c2)
		
def do_mutation(s, m, n):
	i, j = min(m, n), max(m, n)
	s1 = s.copy()
	while i < j:
		s1[i], s1[j] = s1[j], s1[i]
		i += 1
		j -= 1
	return s1
	
def compute_fitness(G, s):
	l = 0
	for i in range(len(s)-1):
		l += G[s[i]][s[i+1]]
	l += G[s[len(s)-1]][s[0]]	
	return l
	
def get_elite(G, gen, k):
	gen = sorted(gen, key=lambda s: compute_fitness(G, s))
	#print(gen)
	return gen[:k]
		
def TSP_GA(P, G, OPT=None, ntrial = 2000, k=50):
	n_p = k
	mutation_prob = 0.1
	gen = []
	path = list(range(len(G)))
	while len(gen) < n_p:
		path1 = path.copy()
		np.random.shuffle(path1)
		if not path1 in gen:
			gen.append(path1)	
	
	gens = list(range(ntrial))
	best_fits = [float('NaN')]*ntrial
	bf = np.inf
	#mean_fits = [float('NaN')]*ntrial
	#sd_fits = [float('NaN')]*ntrial
	for trial in range(ntrial):
		gen = get_elite(G, gen, k)
		gen_costs = [(round(compute_fitness(G, s),3), s) for s in gen]
		#mean_fits[trial] = np.mean(gen_costs)[0]
		#sd_fits[trial] = np.std(gen_costs)[0]
		best_fits[trial] = gen_costs[0][0]
		if best_fits[trial] < bf:
			bf = best_fits[trial]
			plot_GA(P, gen_costs[0][1], trial, [x[0] for x in gen_costs], gens, best_fits) #mean_fits, sd_fits)
		#plot_GA_OPT(P, gen_costs[0][1], OPT, trial, [x[0] for x in gen_costs], gens, best_fits) #mean_fits, sd_fits)
		next_gen = []
		for i in range(len(gen)):
			for j in range(i+1, len(gen)):
				c1, c2 = do_crossover(gen[i], gen[j], np.random.randint(0, len(gen[i])))
				next_gen.append(c1)
				next_gen.append(c2)
			if np.random.rand() < mutation_prob:
				m = np.random.randint(0, len(gen[i]))
				while True:
					n = np.random.randint(0, len(gen[i]))
					if m != n:
						break
				c = do_mutation(gen[i], m, n)
				next_gen.append(c)
		gen = next_gen

import seaborn as sns

def plot_GA(P, path, gen_num, gen_costs, gens, best_fits): #mean_fits, sd_fits):
	global id
	plt.rcParams.update({'font.size': 20})
	print(gen_num)
	plt.figure(figsize=(15,20))
	plt.subplot(211)
	plt.plot([x[0] for x in P], [x[1] for x in P], 'r.', markersize=20)
	for i, p in enumerate(P):
		plt.annotate(str(i), (p[0], p[1]))
	path = path + [path[0]]
	print(path)
	gen_cost = 0
	for i in range(len(path)-1):
		gen_cost += dist(P, path[i], path[i+1])
	plt.plot([P[x][0] for x in path], [P[x][1] for x in path], 'g-', label='GA', lw=5, alpha=0.4)
	plt.legend()
	plt.title('Best Chromosome cost = {:.03f}, Genration = {}'.format(gen_cost, gen_num), size=20)
	plt.grid()
	plt.subplot(212)
	#plt.errorbar(gens, mean_fits, yerr=sd_fits, linestyle='None', marker='^')
	plt.plot(gens, best_fits, 'r.-')
	plt.xlabel('generation', size=20)
	plt.ylabel('best fitness value', size=20)
	plt.grid()
	plt.title('Fitness value over generation', size=20)
	plt.savefig('out/graph_{:03d}.png'.format(id))
	plt.close()
	id += 1
		
def plot_GA_OPT(P, path, OPT, gen_num, gen_costs, gens, best_fits): #mean_fits, sd_fits):
	global id
	plt.rcParams.update({'font.size': 20})
	print(gen_num)
	plt.figure(figsize=(15,20))
	plt.subplot(311)
	plt.plot([x[0] for x in P], [x[1] for x in P], 'r.', markersize=20)
	for i, p in enumerate(P):
		plt.annotate(str(i), (p[0], p[1]))
	opt_path = sum(OPT, [])
	plt.plot([P[x][0] for x in opt_path], [P[x][1] for x in opt_path], 'b-', label='OPT')
	opt_cost = 0
	for i in range(len(opt_path)-1):
		opt_cost += dist(P, opt_path[i], opt_path[i+1])
	path = path + [path[0]]
	print(path)
	gen_cost = 0
	for i in range(len(path)-1):
		gen_cost += dist(P, path[i], path[i+1])
	plt.plot([P[x][0] for x in path], [P[x][1] for x in path], 'g-', label='GA', lw=5, alpha=0.4)
	plt.legend()
	plt.title('Best Chromosome cost = {:.03f}, Genration = {}, OPT cost = {:.03f}'.format(gen_cost, gen_num, opt_cost), size=20)
	plt.grid()
	plt.subplot(312)
	if min(gen_costs) != max(gen_costs):
		plt.hist(gen_costs, bins=np.linspace(min(gen_costs), max(gen_costs), 100))
	else:
		plt.hist(gen_costs)
	plt.xlabel('fitness value', size=20)
	plt.ylabel('count', size=20)
	plt.grid()
	plt.title('Distribution of fitness values in elite chromosomes', size=20)
	plt.subplot(313)
	#plt.errorbar(gens, mean_fits, yerr=sd_fits, linestyle='None', marker='^')
	plt.plot(gens, best_fits, 'r.-')
	plt.xlabel('generation', size=20)
	plt.ylabel('best fitness value', size=20)
	plt.grid()
	plt.title('Fitness value over generation', size=20)
	plt.savefig('out/graph_{:03d}.png'.format(id))
	plt.close()
	id += 1


def dist(P, i, j):
	return np.sqrt((P[i][0]-P[j][0])**2+(P[i][1]-P[j][1])**2)
	
def BTSP(P):
	#print(P)
	n = len(P)
	D = np.ones((n,n))*np.inf
	path = np.ones((n,n), dtype=int)*(-1)
	D[n-2,n-1] = dist(P, n-2, n-1)
	path[n-2,n-1] = n-1
	for i in range(n-3,-1,-1):
		m = np.inf
		for k in range(i+2,n):
			if m > D[i+1,k] + dist(P,i,k):
				m, mk = D[i+1,k] + dist(P,i,k), k
		D[i,i+1] = m
		path[i,i+1] = mk
		for j in range(i+2,n):
			D[i,j] = D[i+1,j] + dist(P,i,i+1)
			path[i,j] = i+1	
			#plot_DP(P, D, path)
	D[0,0] = D[0,1] + dist(P,0,1)
	path[0,0] = 1
	return D, path
	
def get_tsp(path, i, j, n):
	if n < 0:
		return []
	if i <= j:
		k = path[i,j]
		return [k] + get_tsp(path, k, j, n-1)
	else:
		k = path[j,i]
		return get_tsp(path, i, k, n-1) + [k]

import seaborn as sns		
def plot_DP(P, D, path):
	global id
	plt.rcParams.update({'font.size': 20})
	plt.figure(figsize=(20,20))
	plt.subplot(211)
	plt.plot([x[0] for x in P], [x[1] for x in P], 'g.') #, markersize=20)
	for i in range(len(D)):
		for j in range(len(D)):
			if path[i,j] != -1:
				p, q = P[i], P[path[i, j]]
				plt.arrow(p[0], p[1], q[0]-p[0], q[1]-p[1], color='r', shape='full', lw=0, length_includes_head=True, head_width=.01)
	plt.title('Bitonic TSP DP backpointers', size=20)
	plt.grid()
	plt.subplot(212), sns.heatmap(D, annot=True, fmt=".1f", linewidths=.1), plt.title('Bitonic TSP DP table', size=20)
	plt.savefig('out/graph_{:03d}.png'.format(id))
	plt.close()
	id += 1
		
def plot_BTSP(P, path):
	global id
	plt.figure(figsize=(20,20))
	plt.plot([x[0] for x in P], [x[1] for x in P], 'g.') #, markersize=20)
	#i, j = 0, 0
	#l = 0
	#p = [i]
	#while l < n:
	#	if i <= j:
	#		k = path[i,j]
	#		i = k
	#	else:
	#		k = path[j,i]
	#		j = k
	#	p += [k]
	#	l += 1
	#p += [0]
	
	p = get_tsp(path, 0, 0, len(path))
	p = [0] + p + [0]
	print(p)
	for i in range(len(p)-1):
		plt.arrow(P[p[i]][0], P[p[i]][1], P[p[i+1]][0]-P[p[i]][0], P[p[i+1]][1]-P[p[i]][1], color='r', shape='full', lw=0, length_includes_head=True, head_width=.01)
	plt.grid()
	plt.savefig('out/graph_{:03d}.png'.format(id))
	plt.close()
	id += 1
	
	
def dfs(adj, x):
	#write your code here
	visited = [False]*len(adj)
	stack = [x]
	visited[x] = True
	path = []
	parent = {}
	while len(stack) > 0:
		u = stack.pop(-1)
		path.append(u)
		#if parent.get(u, None) != None:
		#	path.append(parent[u])
		for v in adj[u]:
			if not visited[v]:
				stack.append(v)	
				parent[v] = u
				visited[v] = True
	return path

import queue	
def mst(adj, OPT):
	inf = np.inf
	c = [inf]*n
	s = 0
	c[s] = 0
	visited = [False]*n
	parent = [None]*n
	h = queue.PriorityQueue()
	for v in range(n):
		h.put((c[v], v))
	edges = []
	while not h.empty():
		w, u = h.get()
		if visited[u]: continue
		visited[u] = True
		if parent[u] != None:
			edges.append((parent[u], u))
			plot_MST_OPT(P, edges, OPT)
		for v in range(n):
			if v == u: continue
			if (not visited[v]) and (c[v] > adj[u][v]):
				c[v] = adj[u][v]
				parent[v] = u
				h.put((c[v], v))
	adj = [[] for _ in range(n)]
	#root = None
	spanning_tree = []
	for i in range(n):
		spanning_tree.append((i, parent[i]))
		if parent[i] != None:
			adj[parent[i]].append(i)
		#else:
		#	root = i
	#print(spanning_tree)
	path = dfs(adj, 0) #root)
	path = path + [path[0]]
	edges = []
	vertices = set([])
	for i in range(len(path)-1):
		if not path[i+1] in vertices:
			vertices.add(path[i+1])
			edges.append((path[i], path[i+1]))
	print('here', path)
	plot_MST_OPT(P, edges, OPT, 'MST_TSP')
	return path
		
def plot_MST_OPT(P, edges, OPT, mst_label='MST'):
	global id
	plt.rcParams.update({'font.size': 20})
	plt.figure(figsize=(20,20))
	plt.plot([x[0] for x in P], [x[1] for x in P], 'r.', markersize=20)
	for i, p in enumerate(P):
		plt.annotate(str(i), (p[0], p[1]))
	path = sum(OPT, [])
	plt.plot([P[x][0] for x in path], [P[x][1] for x in path], 'b-', label='OPT')
	opt_cost = 0
	for i in range(len(path)-1):
		opt_cost += dist(P, path[i], path[i+1])
	mst_cost = 0
	u, v = edges[0]
	plt.plot([P[u][0], P[v][0]], [P[u][1], P[v][1]], 'g-', label=mst_label, lw=5, alpha=0.5)
	for u, v in edges[1:]:
		plt.plot([P[u][0], P[v][0]], [P[u][1], P[v][1]], 'g-', lw=5, alpha=0.5)
		mst_cost += dist(P, u, v)
	plt.legend()
	plt.title('MST {} Cost = {:.03f}, OPT cost = {:.03f}'.format('' if mst_label=='MST' else '+ DFS TSP', mst_cost, opt_cost) + ('' if mst_label == 'MST' else ', Approx. Ratio={:.02f}'.format(mst_cost/opt_cost)), size=20)
	plt.grid()
	plt.savefig('out/graph_{:03d}.png'.format(id))
	plt.close()
	id += 1

if __name__ == '__main__':
	'''
	n = 20 #100 # 20
	P = []
	for i in range(n):
		P.append((np.random.rand(), np.random.rand()))
	n = len(P)
	adj = [[0 for _ in range(n)] for _ in range(n)]
	for i in range(n):
		for j in range(i+1, n):
			adj[i][j] = adj[j][i] = dist(P, i, j)
	#print(adj)
	opt_cost, opt_path, _, _ = TSP2(adj)
	#TSP_GA(P, adj)#, opt_path)
	#TSP_SA(adj, P)
	#print('OPT', opt_cost)
	
	path = mst(adj, opt_path)
	print(path)
	cost = sum([adj[path[i]][path[i+1]] for i in range(n-1)])
	#print(opt_path)
	print(cost, opt_cost, cost / opt_cost)
	
	P = sorted(P)
	D, path = BTSP(P)
	print(path)
	plot_BTSP(P, path)
	#print_answer(*TSP(read_data()))
	#gen_TSP_data(5)
	'''
	
	N = 100 # 10
	for n in range(3, N):
		print(n)
		m = n*(n-1)//2
		graph = [[INF] * n for _ in range(n)]
		for u in range(n):
			graph[u][u] = 0
			for v in range(u+1, n):
				graph[u][v] = graph[v][u] = np.random.randint(10, 100)
		#TSP_SA(graph)
		solI, pathI, tI, sol_type = TSP2(graph)
		print(pathI, sol_type)
		#sol, path, t = TSP(graph)
		#draw_graph2(graph, sol, path, t, solI, pathI, tI, sol_type)
		draw_graph3(graph, solI, pathI, tI)
	
	#print(times, timesI)
	plt.figure(figsize=(10,10))
	#plt.scatter(range(3,21), times, color='red', s=50)
	#plt.plot(range(3,21), times, 'r.-', markersize=20, label='DP')
	x, y = list(range(3,N)), timesI
	print(sol_types)
	for i in range(len(x)):
		col = 'g' if sol_types[i] == 'OPTIMAL' else 'b' 
		plt.scatter(x[i], y[i], color=col, s=50) #, label='ILP')
	#plt.legend()
	plt.grid()
	plt.xticks(range(3,N,2), range(3,N,2))
	plt.xlabel('n', size=10)
	plt.ylabel('time (in seconds) taken', size=10)
	plt.title('TSP with Integer Linear Program', size=20) #Dynamic programming and 
	plt.show()
	