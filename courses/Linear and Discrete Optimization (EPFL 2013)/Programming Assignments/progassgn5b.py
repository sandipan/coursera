# conduct a breadth-first search of G starting from s
def bfs(G, s):
    d = dict()
    p = dict()
    for v in G:
        d[v] = None
        p[v] = None
    d[s] = 0

    queue = [s]
    while queue:
        u = queue.pop(0)
        for v in G[u]:
            if d[v] is None:
                d[v] = d[u] + 1
                p[v] = u
                queue.append(v)
    return d, p

# returns a bipartition of an undirected graph G if it exists
def bipartite(G):
    A, B = [], []

    d = dict()
    for v in G:
        d[v] = None

    for s in G:
        if d[s] is not None:
            continue

        d[s] = 0

        queue = [s]
        while queue:
            u = queue.pop(0)

            if d[u] % 2 == 0:
                A.append(u)
            else:
                B.append(u)

            for v in G[u]:
                if d[v] is None:
                    d[v] = d[u] + 1
                    queue.append(v)
                elif (d[u] + d[v]) % 2 == 0:
                    return None, None

    return A, B

if __name__ == '__main__':

    # small example
    G = {'a': ['b', 'c', 'g'],
         'b': ['a', 'd', 'e'],
         'c': ['a', 'd'],
         'd': ['b', 'c'],
         'e': ['b'],
         'f': [],
         'g': ['a'],}
    s = 'a'

    d, p = bfs(G, s)
    print d
    print p

    print bipartite(G)
