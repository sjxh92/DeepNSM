import numpy as np
import networkx as nx
from MetroNetwork import NetworkEnvironment as env


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