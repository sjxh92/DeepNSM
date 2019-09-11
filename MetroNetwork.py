import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from Request_generate import RequestGenerate

NODE_NUM = 13
LINK_NUM = 22
LAMDA = 3 / 10
J_NODE = 3
M_VM = 3
K_LINK = 3
W_WAVELENGTH = 3
NODE_CAPACITY = 100
LINK_CAPACITY = 25

COMPUTING_REQUIREMENT = 5
BANDWIDTH_REQUIREMENT = 2


class NetworkEnvironment(object):
    topology = None

    def __init__(self):
        # self.n_feature = NODE_NUM + 2 + J_NODE * M_VM + K_LINK * W_WAVELENGTH

        # node capacity + link capacity + request node + request traffic + holding time
        self.n_feature = NODE_NUM + LINK_NUM + 1 + 1
        self.action_space = []
        self._topology()
        self.init_action_space()
        self.n_action = len(self.action_space)
        self.request = RequestGenerate(NODE_NUM)
        self.X = 0
        self.memory = np.zeros([0, 4], dtype=int)  # nodeid, traffic, starttime, endtime
        self.topology = nx.Graph()

    def _topology(self):
        for i in range(NODE_NUM):
            self.topology.add_node(i, capacity=100)
        self.topology.add_edge(0, 1, weight=1050, capacity=LINK_CAPACITY)
        self.topology.add_edge(1, 0, weight=1050, capacity=LINK_CAPACITY)
        self.topology.add_edge(1, 2, weight=600, capacity=LINK_CAPACITY)
        self.topology.add_edge(2, 1, weight=600, capacity=LINK_CAPACITY)
        self.topology.add_edge(0, 2, weight=1500, capacity=LINK_CAPACITY)
        self.topology.add_edge(2, 0, weight=1500, capacity=LINK_CAPACITY)
        self.topology.add_edge(1, 3, weight=750, capacity=LINK_CAPACITY)
        self.topology.add_edge(3, 1, weight=750, capacity=LINK_CAPACITY)
        self.topology.add_edge(3, 4, weight=600, capacity=LINK_CAPACITY)
        self.topology.add_edge(4, 3, weight=600, capacity=LINK_CAPACITY)
        self.topology.add_edge(4, 5, weight=1200, capacity=LINK_CAPACITY)
        self.topology.add_edge(5, 4, weight=1200, capacity=LINK_CAPACITY)
        self.topology.add_edge(4, 6, weight=600, capacity=LINK_CAPACITY)
        self.topology.add_edge(6, 4, weight=600, capacity=LINK_CAPACITY)
        self.topology.add_edge(6, 7, weight=750, capacity=LINK_CAPACITY)
        self.topology.add_edge(7, 6, weight=750, capacity=LINK_CAPACITY)
        self.topology.add_edge(7, 8, weight=750, capacity=LINK_CAPACITY)
        self.topology.add_edge(8, 7, weight=750, capacity=LINK_CAPACITY)
        self.topology.add_edge(8, 9, weight=750, capacity=LINK_CAPACITY)
        self.topology.add_edge(9, 8, weight=750, capacity=LINK_CAPACITY)
        self.topology.add_edge(6, 9, weight=1350, capacity=LINK_CAPACITY)
        self.topology.add_edge(9, 6, weight=1350, capacity=LINK_CAPACITY)
        self.topology.add_edge(2, 5, weight=1800, capacity=LINK_CAPACITY)
        self.topology.add_edge(5, 2, weight=1800, capacity=LINK_CAPACITY)
        self.topology.add_edge(5, 9, weight=1050, capacity=LINK_CAPACITY)
        self.topology.add_edge(9, 5, weight=1050, capacity=LINK_CAPACITY)
        self.topology.add_edge(3, 10, weight=1950, capacity=LINK_CAPACITY)
        self.topology.add_edge(10, 3, weight=1950, capacity=LINK_CAPACITY)
        self.topology.add_edge(0, 7, weight=2400, capacity=LINK_CAPACITY)
        self.topology.add_edge(7, 0, weight=2400, capacity=LINK_CAPACITY)
        self.topology.add_edge(10, 11, weight=600, capacity=LINK_CAPACITY)
        self.topology.add_edge(11, 10, weight=600, capacity=LINK_CAPACITY)
        self.topology.add_edge(10, 12, weight=750, capacity=LINK_CAPACITY)
        self.topology.add_edge(12, 10, weight=750, capacity=LINK_CAPACITY)
        self.topology.add_edge(12, 13, weight=150, capacity=LINK_CAPACITY)
        self.topology.add_edge(13, 12, weight=150, capacity=LINK_CAPACITY)
        self.topology.add_edge(5, 13, weight=1800, capacity=LINK_CAPACITY)
        self.topology.add_edge(13, 5, weight=1800, capacity=LINK_CAPACITY)
        self.topology.add_edge(11, 13, weight=300, capacity=LINK_CAPACITY)
        self.topology.add_edge(13, 11, weight=300, capacity=LINK_CAPACITY)
        self.topology.add_edge(8, 12, weight=300, capacity=LINK_CAPACITY)
        self.topology.add_edge(12, 8, weight=300, capacity=LINK_CAPACITY)
        self.topology.add_edge(8, 11, weight=1050, capacity=LINK_CAPACITY)
        self.topology.add_edge(11, 8, weight=1050, capacity=LINK_CAPACITY)
        self.topology.add_edge(0, 1, weight=1050, capacity=LINK_CAPACITY)
        self.topology.add_edge(0, 1, weight=1050, capacity=LINK_CAPACITY)


if __name__ == "__main__":
    TP = NetworkEnvironment()
    print(TP.topology.nodes[2]['capacity'])
    print(TP.topology[0][1])
    print(len(TP.topology.edges))
    a = [[0, 7, 6],
         [0, 7, 6],
         [0, 7, 6]]
    print(a)
    a[0].append(0)
    print(a)
    a[1].append(7)
    print(a)
    a[2].append(6)
    print(a)

    # for u, v, d in TP.topology.edges(data='capacity'):
    # print((u, v, d))
    # print()
