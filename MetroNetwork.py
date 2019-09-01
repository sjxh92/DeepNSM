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

    def _topology(self):
        self.topology = nx.Graph()
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

    def init_action_space(self):
        for i in range(len(self.topology.nodes)):
            if i != 6:
                # k_paths = nx.all_shortest_paths(self.topology, i, 6, method='dijkstra')
                k_paths1 = nx.all_simple_paths(self.topology, i, 6, cutoff=3)
                list_paths = list(k_paths1)
                for j in range(len(list_paths)):
                    action_list = []
                    for m in range(len(list_paths[j])):
                        link = list_paths[j]
                        action_list.append(link.copy())
                    for n in range(len(list_paths[j])):
                        node = list_paths[j][n]
                        action_list[n].append(node)
                        self.action_space.append(action_list[n])
        # print(self.action_space)
        return self.action_space

    def get_state(self):
        node = np.empty((1, NODE_NUM))
        link = np.empty((1, LINK_NUM))
        for i in range(NODE_NUM):
            node[0][i] = self.topology.nodes[i]['capacity']

        index = 0
        for u, v, d in self.topology.edges(data='capacity'):
            link[0][index] = d
            index = index + 1

        index, traffic, time = self.request.poissonNext(self.X, LAMDA)
        traffic = traffic[np.newaxis, :]
        index = node[np.newaxis, :]
        time = time[np.newaxis, :]

        s = np.vstack((node, link, traffic, index, time))
        return s

    def update_network(self):
        memory_slice = self.memory[..., 2]
        delete = np.where(memory_slice == self.X)
        for i in delete:

    def step(self, action):

        s = self.get_state()
        b_not_enough = False
        c_not_enough = False
        wrong_action = False
        reward = 0

        current_action = self.action_space[action]
        candidate_node = current_action[-1]

        if s[0][-3] != -1:

        # check the request position
        if current_action[0] == s[0][-2]:  # corresponding to index in u
            # check computation resource
            if self.topology.nodes[candidate_node]['capacity'] >= COMPUTING_REQUIREMENT:
                # check the bandwidth resource
                for i in range(len(current_action) - 3):
                    if self.topology[current_action[i]][current_action[i + 1]]['capacity'] < BANDWIDTH_REQUIREMENT:
                        b_not_enough = True
                        break
            else:
                c_not_enough = True
        else:
            wrong_action = True

        # update the reward and indicator and map the request
        # reward = np.array().shape(1, 1)
        if b_not_enough or c_not_enough:
            reward = -1
        if wrong_action:
            reward = -2
        if not b_not_enough and not c_not_enough and not wrong_action:
            reward = 1
            self.topology.nodes[candidate_node]['capacity'] -= COMPUTING_REQUIREMENT
            for i in range(len(current_action) - 3):
                self.topology[current_action[i]][current_action[i + 1]]['capacity'] -= BANDWIDTH_REQUIREMENT
            np.append(self.memory, [[index, traffic, self.X, self.X + time]], axis=0)
        # update the next state (including network state and request)
        self.X = self.X + 1
        s_ = self.get_state()
        return s_, reward, self.X

    def show_topology(self):
        print(self.topology.nodes.data())
        print(self.topology.edges.data())


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
