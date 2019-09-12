import networkx as nx
import matplotlib.pyplot as plt

G = nx.cycle_graph(7)
paths = list(nx.shortest_simple_paths(G, 0, 3))
print(paths)
# G = nx.path_graph(5, create_using=nx.DiGraph())
nx.draw(G, with_labels=True)
plt.title('有向图')
plt.axis('on')
plt.xticks([])
plt.yticks([])
# plt.show()

FG = nx.Graph()
FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.275)])
for n, nbrs in FG.adjacency():
    print(n, nbrs.items())
    for nbr, eattr in nbrs.items():
        data = eattr['weight']
        print('(%d, %d, %0.3f)' % (n, nbr, data))
