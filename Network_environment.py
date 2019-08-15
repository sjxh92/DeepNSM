import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import Request_generate as rg

NODE_NUM = 6
LINK_NUM = 8
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
        self.n_feature = NODE_NUM + 2 + J_NODE * M_VM + K_LINK * W_WAVELENGTH
        self.n_action_admission = 2
        self.n_action_mapping = J_NODE * M_VM + K_LINK * W_WAVELENGTH

    def _topology(self):
        self.topology = nx.Graph()
        for i in range(NODE_NUM):
            self.topology.add_node(i, vm=np.array([VM_CAPACITY, VM_CAPACITY, VM_CAPACITY, VM_CAPACITY, VM_CAPACITY, VM_CAPACITY]))
        self.topology.add_edge(1, 3, weight=12, wavelength=np.array([W_CAPACITY, W_CAPACITY, W_CAPACITY]))
        self.topology.add_edge(3, 5, weight=6, wavelength=np.array([W_CAPACITY, W_CAPACITY, W_CAPACITY]))
        self.topology.add_edge(5, 4, weight=5, wavelength=np.array([W_CAPACITY, W_CAPACITY, W_CAPACITY]))
        self.topology.add_edge(4, 2, weight=15, wavelength=np.array([W_CAPACITY, W_CAPACITY, W_CAPACITY]))
        self.topology.add_edge(2, 0, weight=7, wavelength=np.array([W_CAPACITY, W_CAPACITY, W_CAPACITY]))

        self.topology.add_edge(1, 2, weight=5, wavelength=np.array([W_CAPACITY, W_CAPACITY, W_CAPACITY]))
        self.topology.add_edge(3, 4, weight=10, wavelength=np.array([W_CAPACITY, W_CAPACITY, W_CAPACITY]))
        self.topology.add_edge(1, 6, weight=5, wavelength=np.array([W_CAPACITY, W_CAPACITY, W_CAPACITY]))
        self.topology.add_edge(3, 6, weight=8, wavelength=np.array([W_CAPACITY, W_CAPACITY, W_CAPACITY]))
        self.topology.add_edge(4, 6, weight=7, wavelength=np.array([W_CAPACITY, W_CAPACITY, W_CAPACITY]))
        self.topology.add_edge(2, 6, weight=9, wavelength=np.array([W_CAPACITY, W_CAPACITY, W_CAPACITY]))

    def next(self, action):

        s =

    def _show_topology(self):
        print(self.topology.nodes.data())
        print(self.topology.edges.data())


if __name__ == "__main__":
    TP = NetworkEnvironment()
    print(TP.topology.nodes.data())
    print(TP.topology.edges.data())
