import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import Request_generate as rg

NODE_NUM = 7
LINK_NUM = 12
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
        self.n_feature = NODE_NUM + LINK_NUM + 1
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
        self._topology()

    def _topology(self):
        self.topology = nx.Graph()
        for i in range(NODE_NUM):
            self.topology.add_node(i, capacity=100)
        self.topology.add_edge(0, 1, distance=5, capacity=LINK_CAPACITY)
        self.topology.add_edge(1, 3, distance=12, capacity=LINK_CAPACITY)
        self.topology.add_edge(3, 5, distance=6, capacity=LINK_CAPACITY)
        self.topology.add_edge(5, 4, distance=5, capacity=LINK_CAPACITY)
        self.topology.add_edge(4, 2, distance=15, capacity=LINK_CAPACITY)
        self.topology.add_edge(2, 0, distance=7, capacity=LINK_CAPACITY)

        self.topology.add_edge(1, 2, distance=5, capacity=LINK_CAPACITY)
        self.topology.add_edge(3, 4, distance=10, capacity=LINK_CAPACITY)
        self.topology.add_edge(1, 6, distance=5, capacity=LINK_CAPACITY)
        self.topology.add_edge(3, 6, distance=8, capacity=LINK_CAPACITY)
        self.topology.add_edge(4, 6, distance=7, capacity=LINK_CAPACITY)
        self.topology.add_edge(2, 6, distance=9, capacity=LINK_CAPACITY)

    def init_state(self):
        r = random.randint(0, 6)
        s = np.empty((1, self.n_feature))
        for i in range(NODE_NUM):
            s[0][i] = self.topology.nodes[i]['capacity']

        index = NODE_NUM
        for u, v, d in self.topology.edges(data='capacity'):
            s[0][index] = d
            index = index + 1
        s[0][self.n_feature - 1] = r
        return s

    def next(self, action):

        s = self.init_state()
        b_not_enough = False
        c_not_enough = False
        wrong_action = False
        reward = 0

        current_action = self.action_space[action]
        candidate_node = current_action[-1]

        # check the request position
        if current_action[0] == s[0][-1]:
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

        # update the next state (including network state and request)
        r = random.randint(0, 6)
        s_ = np.empty((1, self.n_feature))
        for i in range(NODE_NUM):
            s_[0][i] = self.topology.nodes[i]['capacity']

        index = NODE_NUM
        for u, v, d in self.topology.edges(data='capacity'):
            s_[0][index] = d
            index = index + 1
        s_[0][self.n_feature - 1] = r
        return s_, reward

    def show_topology(self):
        print(self.topology.nodes.data())
        print(self.topology.edges.data())


if __name__ == "__main__":
    TP = NetworkEnvironment()
    print(TP.topology.nodes[2]['capacity'])
    print(TP.topology[0][1])

    # for u, v, d in TP.topology.edges(data='capacity'):
    # print((u, v, d))
    # print()
