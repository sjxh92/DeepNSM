import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import Request_generate as rg

NODE_NUM = 7
LINK_NUM = 12
J_NODE = 3
M_VM = 3
K_LINK = 3
W_WAVELENGTH = 3
VM_CAPACITY = 100
W_CAPACITY = 10


class NetworkEnvironment:
    def __init__(self):
        self._topology()
        self.request = rg
        # self.n_feature = NODE_NUM + 2 + J_NODE * M_VM + K_LINK * W_WAVELENGTH
        self.n_feature = NODE_NUM + LINK_NUM + 1
        self.n_action_admission = 2
        self.action_space = np.array([[0, 1, 6, 0],
                                      [0, 1, 6, 1],
                                      [0, 1, 6, 6],
                                      [0, 2, 6, 0],
                                      [0, 2, 6, 2],
                                      [0, 2, 6, 6],
                                      [1, 6, 1],
                                      [1, 6, 6],
                                      [1, 2, 6, 1],
                                      [1, 2, 6, 2],
                                      [1, 2, 6, 6],
                                      [2, 6, 2],
                                      [2, 6, 6],
                                      [3, 6, 3],
                                      [3, 6, 6],
                                      [4, 6, 4],
                                      [4, 6, 6],
                                      [5, 3, 6, 5],
                                      [5, 3, 6, 3],
                                      [5, 3, 6, 6],
                                      [5, 4, 6, 5],
                                      [5, 4, 6, 4],
                                      [5, 4, 6, 6]])
        self.n_action_mapping = len(self.action_space)


    def _topology(self):
        self.topology = nx.Graph()
        for i in range(NODE_NUM):
            self.topology.add_node(i, capacity=100)
        self.topology.add_edge(1, 3, weight=12, capacity=10)
        self.topology.add_edge(3, 5, weight=6, capacity=10)
        self.topology.add_edge(5, 4, weight=5, capacity=10)
        self.topology.add_edge(4, 2, weight=15, capacity=10)
        self.topology.add_edge(2, 0, weight=7, capacity=10)

        self.topology.add_edge(1, 2, weight=5, capacity=10)
        self.topology.add_edge(3, 4, weight=10, capacity=10)
        self.topology.add_edge(1, 6, weight=5, capacity=10)
        self.topology.add_edge(3, 6, weight=8, capacity=10)
        self.topology.add_edge(4, 6, weight=7, capacity=10)
        self.topology.add_edge(2, 6, weight=9, capacity=10)

    def _state_transformation(self):
        r = 0
        s = np.empty((self.n_feature, 1))
        for i in range(self.n_feature):
            if i < NODE_NUM:
                s[i] = self.topology.nodes[i]['capacity']
            elif i < NODE_NUM + LINK_NUM:
                s[i] = self.topology.edges[i]['capacity']
            else:
                s[i] = r
        print(s)

    def next(self, action):

        s = 0

    def _show_topology(self):
        print(self.topology.nodes.data())
        print(self.topology.edges.data())


if __name__ == "__main__":
    TP = NetworkEnvironment()
    print(TP.topology.nodes[2]['capacity'])
    print(TP.topology.edges[1])
    # TP._state_transformation()
    # print()
