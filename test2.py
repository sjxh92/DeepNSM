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
plt.show()
