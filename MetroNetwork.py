import numpy as np
import pandas as pd
import networkx as nx
from itertools import islice
import random
import matplotlib.pyplot as plt

NODE_NUM = 7
LINK_NUM = 12
LAMDA = 3 / 10
J_NODE = 3
M_VM = 3
K_LINK = 3
W_WAVELENGTH = 3

COMPUTING_REQUIREMENT = 5
BANDWIDTH_REQUIREMENT = 2


class NetworkEnvironment(nx.Graph):

    def __init__(self, **attr):
        # self.n_feature = NODE_NUM + 2 + J_NODE * M_VM + K_LINK * W_WAVELENGTH

        # node utilization + link utilization + request node + request traffic + holding time
        super().__init__(**attr)
        self.n_feature = NODE_NUM + LINK_NUM + 1 + 1
        self.action_space = []
        self.n_action = len(self.action_space)
        self.X = 0
        self.memory = np.zeros([0, 4], dtype=int)  # nodeid, traffic, starttime, endtime
        self._topology()

    def _topology(self):
        capacity = np.zeros(shape=(50, 10))
        for i in range(50):
            capacity[i] = i

        for i in range(NODE_NUM):
            self.add_node(i, utilization=capacity)
        self.add_edge(0, 1, weight=1050, utilization=capacity)
        self.add_edge(1, 0, weight=1050, utilization=capacity)
        self.add_edge(1, 2, weight=600, utilization=capacity)
        self.add_edge(2, 1, weight=600, utilization=capacity)
        self.add_edge(2, 3, weight=1500, utilization=capacity)
        self.add_edge(3, 2, weight=1500, utilization=capacity)
        self.add_edge(3, 4, weight=750, utilization=capacity)
        self.add_edge(4, 3, weight=750, utilization=capacity)
        self.add_edge(4, 5, weight=600, utilization=capacity)
        self.add_edge(5, 4, weight=600, utilization=capacity)
        self.add_edge(5, 0, weight=1200, utilization=capacity)
        self.add_edge(0, 5, weight=1200, utilization=capacity)
        self.add_edge(1, 5, weight=1200, utilization=capacity)
        self.add_edge(5, 1, weight=1200, utilization=capacity)
        self.add_edge(2, 4, weight=1200, utilization=capacity)
        self.add_edge(4, 2, weight=1200, utilization=capacity)
        self.add_edge(1, 6, weight=1200, utilization=capacity)
        self.add_edge(6, 1, weight=1200, utilization=capacity)
        self.add_edge(5, 6, weight=1200, utilization=capacity)
        self.add_edge(6, 5, weight=1200, utilization=capacity)
        self.add_edge(2, 6, weight=1200, utilization=capacity)
        self.add_edge(6, 2, weight=1200, utilization=capacity)
        self.add_edge(4, 6, weight=1200, utilization=capacity)
        self.add_edge(6, 4, weight=1200, utilization=capacity)

    def connection_mapping(self, arrivalEvent, path):
        traffic = arrivalEvent[2]
        topo = self.net.topology
        for n, nbrs in topo.adjacency():
            for nbr, attr in nbrs.items():
                for link in path:
                    if n == link[0] and nbr == link[1]:
                        topo[n][nbr]["capacity"] -= traffic
        connection = pd.DataFrame({'id': 1,
                                   'path': path,
                                   'wavelength': None,
                                   'traffic': traffic,
                                   'ta': arrivalEvent[0],
                                   'th': arrivalEvent[1]})
        self.connections.append(connection)

    def connection_release(self, cId, departuretime):
        connection = self.connections.loc(self.connections['id'] == cId)
        path = connection['path']
        traffic = connection['traffic']
        topo = self.net.topology
        for n, nbrs in topo.adjacency():
            for nbr, attr in nbrs.items():
                for link in path:
                    if n == link[0] and nbr == link[1]:
                        topo[n][nbr]['capacity'] += traffic
        self.connections = self.connections[~self.connections['id'].isin(cId)]

    def set_wave_state(self, wave_index, nodes: list, state: bool, check: bool = True):
        """
        set the state of a certain wavelength on a path
        :param wave_index:
        :param nodes:
        :param state:
        :param check:
        :return:
        """
        assert len(nodes) >= 2
        start_node = nodes[0]
        for i in range(1, len(nodes)):
            end_node = nodes[i]
            if check:
                assert self.get_edge_data(start_node, end_node)['is_wave_avai'][wave_index] != state
            self.get_edge_data(start_node, end_node)['is_wave_avai'][wave_index] = state
            start_node = end_node


if __name__ == "__main__":
    TP = NetworkEnvironment()
    print(TP.nodes[2])
    print(TP[0][1])
    print(len(TP.edges))
    # a = [[0, 7, 6],
    #      [0, 7, 6],
    #      [0, 7, 6]]
    # print(a)
    # a[0].append(0)
    # print(a)
    # a[1].append(7)
    # print(a)
    # a[2].append(6)
    # print(a)

    # for u, v, d in TP.topology.edges(data='utilization'):
    # print((u, v, d))
    # print()
