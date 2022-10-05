#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

void ex_1()
{
	std::ifstream f("IntegerArray.txt");
	std::vector<int> v;
	char s[32];
	while (f)
	{
		f.getline(s, 32);
		if (!f)
		{
			break;
		}
		v.push_back(atoi(s));
	}
	std::cout << v.size() << std::endl;
	long long inv = 0;
	for (int i = 0; i < v.size(); ++i)
	{
		for (int j = i + 1; j < v.size(); ++j)
		{
			if (v[i] > v[j])
			{
				++inv;
			}
		}
	}
	std::cout << inv << std::endl;
}

enum pivot
{
	first, last, med3
};

int partition( int a[ ], int lower, int upper, pivot pv )
{
	int i, p, q, t ;

	if (pv == last)
	{
		t = a[lower] ;
		a[lower] = a[upper] ;
		a[upper] = t ;
	}
	else if (pv == med3)
	{
		int mid = (lower + upper) / 2;
		int med3;
		if ((a[lower] <= a[mid] && a[mid] <= a[upper]) || (a[lower] >= a[mid] && a[mid] >= a[upper]))
		{
			med3 = mid;
		}
		else if ((a[mid] <= a[lower] && a[lower] <= a[upper]) || (a[mid] >= a[lower] && a[lower] >= a[upper]))
		{
			med3 = lower;
		}
		else
		{
			med3 = upper;
		}
		t = a[lower] ;
		a[lower] = a[med3] ;
		a[med3] = t ;
	}
	
	p = a[lower];
	i = lower + 1;
	for (int j = lower + 1; j <= upper; ++j)
	{
		if (a[j] < p)
		{
			t = a[i];
			a[i] = a[j];
			a[j] = t;
			++i;
		}
	}
	t = a[lower];
	a[lower] = a[i - 1];
	a[i - 1] = t;
	
	return i - 1;
	
	/*p = lower + 1 ;
	q = upper ;
	i = a[lower] ;

	while ( q >= p )
	{
		while ( a[p] < i )
			p++ ;

		while ( a[q] > i )
			q-- ;

		if ( q > p )
		{
			t = a[p] ;
			a[p] = a[q] ;
			a[q] = t ;
		}
	}

	t = a[lower] ;
	a[lower] = a[q] ;
	a[q] = t ;

	return q ; */
	
}

void quicksort ( int a[ ], int lower, int upper, pivot pivot, int& comp )
{
	int i ;
	if ( upper > lower )
	{
		i = partition ( a, lower, upper, pivot) ;
		comp += upper - lower; //+ 1;
		quicksort ( a, lower, i - 1, pivot, comp ) ;
		quicksort ( a, i + 1, upper, pivot, comp ) ;
	}
}

void ex_2()
{
	std::ifstream f("QuickSort.txt");
	int *a = new int[10000], i = 0;
	char s[32];
	while (f)
	{
		f.getline(s, 32);
		if (!f)
		{
			break;
		}
		a[i++] = atoi(s);
	}
	
	int comp = 0;
	
	//quicksort ( a, 0, 9999, first, comp ) ;
	//std::cout << comp << std::endl;
	
	//quicksort ( a, 0, 9999, last, comp ) ;
	//std::cout << comp << std::endl;
	
	//quicksort ( a, 0, 9999, med3, comp ) ;
	//std::cout << comp << std::endl;
	
	//int arr[] = {12, 3, 5, 1, 7, 2, 100, 4};
	//comp = 0;
	//quicksort ( arr, 0, 7, first, comp ) ;
	//quicksort ( arr, 0, 7, last, comp ) ;
	//quicksort ( arr, 0, 7, med3, comp ) ;
	//std::copy(arr, arr + 8, std::ostream_iterator<int>(std::cout, " "));
	//std::cout << std::endl << comp << std::endl;
}

void contract_edge(std::map<int, std::vector<int> >& graph, int u, int v)
{
	int u_v = u < v ? u : v, d_v = u < v ? v : u;
	std::map<int, std::vector<int> >::iterator it_u_v = graph.find(u_v);
	std::map<int, std::vector<int> >::iterator it_d_v = graph.find(d_v);
	std::vector<int> adj_u_v = it_u_v->second, adj_d_v = it_d_v->second;
	//adj_u_v.insert(adj_u_v.end(), adj_d_v.begin(), adj_d_v.end());
	for (int i = 0; i < adj_d_v.size(); ++i)
	{
		if (adj_d_v[i] != u_v)
		{
			adj_u_v.push_back(adj_d_v[i]);
		}
	}
	graph[u_v] = adj_u_v;
	for (std::map<int, std::vector<int> >::iterator it = graph.begin(); it != graph.end(); ++it)
	{
		int v = it->first;
		std::vector<int> adj = it->second;
		for (std::vector<int>::iterator vit = adj.begin(); vit != adj.end(); )
		{
			if (*vit == d_v)
			{
				if (u_v != v)
				{
					*vit = u_v;
					++vit;
				}
				else
				{
					vit = adj.erase(vit);
				}
			}
			else
			{
				++vit;
			}
		}
		it->second = adj;
	}
	graph.erase(it_d_v);
	/*std::cout << std::endl;
	for (std::map<int, std::vector<int> >::const_iterator it = graph.begin(); it != graph.end(); ++it)
	{
		std::cout << it->first << ": ";
		std::copy(it->second.begin(), it->second.end(), std::ostream_iterator<int>(std::cout, " "));
		std::cout << std::endl;
	}*/
}

void run_karger(std::map<int, std::vector<int> >& graph)
{	
	std::map<int, std::vector<int> >::iterator it;
	for (int n = graph.size(); n > 2; --n)
	{
		int u, v;
		std::vector<int> adj_u;
		it = graph.begin();
		for (int off_u = rand() % n, i = 0; i < off_u; ++it, ++i);
		u = it->first;
		adj_u = it->second;
		v = adj_u[rand() % adj_u.size()];
		//std::cout << std::endl << "(u, v): " << u << " " << v << std::endl;
		contract_edge(graph, u, v);
	}
}

void ex_3()
{
	std::ifstream f("kargerAdj.txt");
	std::map<int, std::vector<int> > graph, temp_graph;
	char s[256];
	while (f) 
	{
		f.getline(s, 256);
		if (!f) 
		{
			break;
		}
		std::vector<int> adj_v;
		char *vs = strtok (s, " \t");
		int v = atoi(vs);
		while (vs != 0)
		{
			vs = strtok (0, " \t");
			if (vs != 0)
			{
				adj_v.push_back(atoi(vs));
			}
		}
		graph[v] = adj_v;
	}
	
	int mindeg = 9999;
	for (std::map<int, std::vector<int> >::const_iterator it = graph.begin(); it != graph.end(); ++it)
	{
		std::cout << it->first << ": ";
		std::copy(it->second.begin(), it->second.end(), std::ostream_iterator<int>(std::cout, " "));
		std::cout << std::endl;
		if (mindeg > it->second.size())
		{
			mindeg = it->second.size();
		}
	}
	
	srand(time(NULL));
	int n = graph.size();
	int mincut = 9999;
	for (int i = 0; i < n * n * log10(n) / log10(2); ++i)
	{
		temp_graph.insert(graph.begin(), graph.end());
		run_karger(temp_graph);
		if (mincut > (temp_graph.begin())->second.size())
		{
			mincut = (temp_graph.begin())->second.size();
		}
		std::cout << (temp_graph.begin())->second.size() << std::endl;
		/*std::cout << std::endl << std::endl;
		for (std::map<int, std::vector<int> >::const_iterator it = graph.begin(); it != graph.end(); ++it)
		{
			std::cout << it->first << ": ";
			std::copy(it->second.begin(), it->second.end(), std::ostream_iterator<int>(std::cout, " "));
			std::cout << std::endl;
		}*/
		temp_graph.clear();
	}
	std::cout << "MinDeg: " << mindeg << std::endl;
	std::cout << " MinCutSize: " << mincut << std::endl;
}

void DFS_Visit(std::vector<std::vector<int> >& graph, int u, int& time, std::vector<int>& visited, std::vector<int>& f, int& count)
{
	visited[u] = 1;
	for (int i = 0; i < graph[u].size(); ++i)
	{
		int v = graph[u][i];
		if (!visited[v])
		{
			DFS_Visit(graph, v, time, visited, f, count);
			++count;
		}
	}
	f[u] = time;
	time = time + 1;
}

void DFS(std::vector<std::vector<int> >& graph, std::vector<int>& f)
{
	int n = graph.size();
	std::vector<int> visited(n);
	for (int u = n - 1; u >= 1; --u)
	{
		visited[u] = 0;
	}
	int time = 1;
	for (int u = n - 1; u >= 1; --u)
	{
		if (!visited[u])
		{
			int count = 0;
			DFS_Visit(graph, u, time, visited, f, count);
			std::cout << "Leader: " << u << " Component size = " << count << std::endl;
		}
	}
	for (int i = 1; i < n; ++i)
	{
		std::cout << i << ":  " << f[i] << std::endl;
	}
}

void ex_4()
{
	const int n = 9; //875714;
	std::ifstream f("SCC_.txt");
	std::vector<std::vector<int> > graph(n + 1, std::vector<int>(0)), rev_graph(n + 1, std::vector<int>(0));
	char s[64];
	while (f) 
	{
		f.getline(s, 64);
		if (!f) 
		{
			break;
		}
		int u = atoi(strtok (s, " \t"));
		int v = atoi(strtok (0, " \t"));
		graph[u].push_back(v);
		rev_graph[v].push_back(u);
	}
	
	for (int i = 1; i < rev_graph.size(); ++i)
	{
		std::cout << i << ": ";
		std::copy(rev_graph[i].begin(), rev_graph[i].end(), std::ostream_iterator<int>(std::cout, " "));
		std::cout << std::endl;
	}
	std::vector<int> ft(n + 1);
	DFS(rev_graph, ft);
	std::map<int, int> vmap;
	for (int i = 1; i <= n; ++i)
	{
		vmap[ft[i]] = i;
	}
	graph[7] = graph[1];
	/*for (int i = 1; i < graph.size(); ++i)
	{
		graph[ft[i]] = graph[i];
	}*/
	std::cout << "Here: " << std::endl;
	for (int i = 1; i < graph.size(); ++i)
	{
		std::cout << i << ": ";
		std::copy(graph[i].begin(), graph[i].end(), std::ostream_iterator<int>(std::cout, " "));
		std::cout << std::endl;
	}
	DFS(graph, ft);
}

int main()
{
	//ex_1();
	//ex_2();
	//ex_3();
	ex_4();
	return 0;
}