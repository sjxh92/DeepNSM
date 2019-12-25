import numpy as np
import pandas as pd
from MetroNetwork import NetworkEnvironment as network
from args import args
import pl
import random
import networkx as nx
import chainer
import chainerrl

# there is no request arriving or leaving at this time
NOARRIVAL_NO = 1  # choose no_action
NOARRIVAL_OT = -1  # choose other actions

ARRIVAL_NO = 1
ARRIVAL_NOOP = 1
ARRIVAL_NOOP_OT = 1
ARRIVAL_OP_OT = 1
ARRIVAL_OP_NO = 1

INIT = 0


class Request(object):
    def __init__(self, index: int, src: int, dst: int, arrival_time: int, leave_time: int, traffic: int):
        super(Request, self).__init__()
        self.index = index
        self.src = src
        self.dst = dst
        self.arrival_time = arrival_time
        self.leave_time = leave_time
        self.traffic = traffic

    def add_allocation(self, path: list, wave_index: int):
        self.path = path
        self.wave_index = wave_index


class NSMGame(object):
    def __init__(self, mode: str, wave_num: int, vm_num: int, max_iter: int, rou: float, mu: float, k: int, f: int,
                 weight):
        """

        :param mode:
        :param max_iter: max number of requests
        :param rou:
        :param mu:
        :param k: ksp
        :param f: f candidate node for du/cu placement
        """
        self.network = network("7node_10link", "/home/mario/PycharmProjects/DeepNSM/Resource", wave_num, max_iter,
                               vm_num)
        self.action_space = []
        self.mode = mode
        self.max_iter = max_iter
        self.wave_num = wave_num
        self.rou = rou
        self.mu = mu
        self.k = k
        self.f = f
        self.weight = weight
        self.NO_ACTION = k * f
        self.event_iter = 0
        self.time = 0
        self.event = []
        self.request = {}

    def action_space_gen(self, arrivalEvent):
        # transform the candidate paths into pandas format [path, weight]
        paths = self.service.candidate_paths(arrivalEvent, 3)

        paths_pandas = pd.DataFrame(columns=['path', 'weight'])
        # rank the candidate paths in the descending order of remaining capacity
        for path in paths:
            capacity = 0
            for link in path:
                capacity += link['capacity']
            paths_pandas = paths_pandas.append(pd.DataFrame({'path': [path], 'weight': [capacity]}))
        paths_pandas.sort_values(by='weight', ascending=False)
        return paths_pandas

    def reset(self):

        """
        reset event and environment
        initiate the events
        :return:
        """

        self.event_iter = 0
        self.event = []
        self.time = 0

        base_time = 0
        rand_val = int(random.random() * 1000000000)
        np.random.seed(rand_val)
        for base_index in range(self.max_iter):
            src = random.randint(0, pl.NODE_NUM)
            arrival = np.random.poisson(lam=self.rou) + base_time + 1
            leave = np.random.poisson(lam=self.mu) + arrival + 1
            traffic = random.randint(0, 10)
            self.request[base_index] = Request(base_index, src, '', arrival, leave, traffic)
            self.event.append([arrival, base_index, True])
            self.event.append([leave, base_index, False])
            base_time = arrival
        self.event.sort(key=lambda time: time[0])

        # return the first state
        self.time = self.request[0].arrival_time
        observation = self.get_state_link(self.time)
        reward = INIT
        done = False
        info = None

        return observation, reward, done, info

    def get_state_link(self, time):

        node_num = self.network.number_of_nodes()
        link_num = self.network.number_of_edges()
        request_holding_time = 100

        state = np.zeros(shape=(pl.WINDOW*(node_num+link_num)+request_holding_time, 10))

        # network state:node state
        for i in range(pl.NODE_NUM):
            node = self.network.nodes[i + 1]
            capacity = node['capacity']
            state_window = capacity[time:time + pl.WINDOW]
            state[(pl.WINDOW * i):(pl.WINDOW * (i + 1))] = state_window
            # print(capacity)

        # network state:link state
        j = pl.NODE_NUM
        for n, nbrs in self.network.adjacency():
            for nbr, eattr in nbrs.items():
                capacity = eattr['is_wave_avai']
                state_window = capacity[time:time + pl.WINDOW]
                state[(10 * j):(10 * (j + 1))] = state_window
                j = j + 1

        # network state:request state
        state_request = np.zeros(shape=(10, 10))

        holding_time = 0
        traffic = 0
        for base_index in range(self.max_iter):
            if self.request[base_index].arrival_time == time:
                holding_time = self.request[base_index].leave_time - time
                traffic = self.request[base_index].traffic
                break

        for m in range(holding_time):
            for n in range(traffic):
                state_request[m][n] = 1
        # print('request:', state_request)
        state[(10 * j):(10 * (j + 1))] = state_request

        # print('state:', state)
        return state

    def get_state_path(self, time):
        pass

    def step(self, action) -> [object, float, bool, dict]:

        """
        according to action, interact with environment
        :param action:
        :return:
        """
        if action is -1:
            return np.array([None, None]), 0, True, None
        #  --------------------------------------------------------------
        done = False
        info = False
        # check if there are events (arrival of departure)
        while self.event[self.event_iter][0] == self.time:
            if 

    def exec_action(self, action: int, req: Request) -> float:
        """
        for the given request, execute the action and get the reward
        :param action:
        :param req:
        :return:
        """
        path_list = list(self.k_shortest_paths(req.src, req.dst))
        is_avai, _, _ = self.network.exist_rw_allocation(path_list)
        print("is_avai")
        print(is_avai)
        if action == self.NO_ACTION:
            if is_avai:
                return ARRIVAL_NO
            else:
                return ARRIVAL_NOOP
        else:
            if is_avai:
                route_index = action // (self.k * self.wave_num)
                wave_index = action % (self.k * self.wave_num)
                if self.network.is_allocable(path_list[route_index], wave_index):
                    self.network.set_wave_state(wave_index=wave_index, nodes=path_list[route_index],
                                                state=False, check=True)
                    req.add_allocation(path_list[route_index], wave_index)
                    return ARRIVAL_OP_OT
                else:
                    return ARRIVAL_OP_NO
            else:
                return ARRIVAL_NOOP_OT

    def k_shortest_paths(self, source, target):
        """
        calculate the paths
        :param source:
        :param target:
        :return:
        """
        if source is None:
            return [None]
        paths = nx.shortest_simple_paths(self.network, source, target, weight='weight')
        path_k = []
        index = 0
        for i in paths:
            index += 1
            if index > self.k:
                break
            path_k.append(i)
        # print(path_k)
        # paths_pandas = pd.DataFrame(columns=['path', 'weight'])
        # for path in path_k:
        #     weight = 0
        #     print(path)
        #     for link in path:
        #         print(link)
        #         weight += link['weight']
        #     paths_pandas = paths_pandas.append(pd.DataFrame({'path': [path], 'weight': [weight]}))
        # paths_pandas.sort_values(by='weight', ascending=False)
        return path_k


if __name__ == "__main__":
    game = NSMGame(mode="LINN", wave_num=10, vm_num=10, max_iter=20, rou=0.1, mu=0.1, k=3, f=3, weight=1)
    paths = list(game.k_shortest_paths(1, 6))
    # print(paths)
    # print(game.network.nodes[1]['capacity'])
    # print(game.network.edges[1, 3]['is_wave_avai'])
    game.reset()
    print(game.event)
    # request = Request(1, 1, 6, 1, 4, 1)
    # reward1 = game.exec_action(1, request)
    # print(reward1)
    pass
