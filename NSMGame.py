import numpy as np
import pandas as pd
from MetroNetwork import NetworkEnvironment as network
from args import args
import pl
import random
import networkx as nx
import chainer
import chainerrl
import logging
logging.basicConfig(level=logging.DEBUG)

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


def show_requests():
    for i in game.event:
        print(i)
    for j in game.request:
        print(game.request[j].index,
              game.request[j].src,
              game.request[j].dst,
              game.request[j].arrival_time,
              game.request[j].leave_time,
              game.request[j].traffic)


class Game(object):
    def __init__(self, mode: str, total_time: int, wave_num: int, vm_num: int, max_iter: int, rou: float, mu: float, k: int, f: int,
                 weight):
        """

        :param mode:
        :param max_iter: max number of requests
        :param rou:
        :param mu:
        :param k: ksp
        :param f: f candidate node for du/cu placement
        """
        self.network = network("7node_10link", "/home/mario/PycharmProjects/DeepNSM/Resource",
                               wave_num=wave_num,
                               vm_num=vm_num,
                               total_time=total_time)
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
        print('rand_val:', rand_val)
        np.random.seed(8)
        for base_index in range(self.max_iter):
            src = np.random.randint(1, pl.NODE_NUM)
            dst = src
            while src == dst:
                dst = np.random.randint(1, pl.NODE_NUM)
            arrival = np.random.poisson(lam=self.rou) + base_time + 1
            leave = np.random.poisson(lam=self.mu) + arrival + 1
            traffic = np.random.randint(0, 10)
            self.request[base_index] = Request(base_index, src, dst, arrival, leave, traffic)
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
        print(node_num)
        print(link_num)

        state = np.zeros(shape=(pl.WINDOW*(node_num+2 * link_num)+request_holding_time, 10))
        print('+++++++++++++++', state.shape)
        # network state:node state
        for i in range(pl.NODE_NUM):
            node = self.network.nodes[i + 1]
            capacity = node['capacity']
            state_window = capacity[time:time + pl.WINDOW]
            if state_window.shape[0] < pl.WINDOW:
                append_list = [[0 for _ in range(10)]]
                for j in range(pl.WINDOW - state_window.shape[0]):
                    state_window = np.concatenate((state_window, append_list), axis=0)
            state[(pl.WINDOW * i):(pl.WINDOW * (i + 1))] = state_window
            # print(capacity)

        # network state:link state
        j = pl.NODE_NUM
        for n, nbrs in self.network.adjacency():
            for nbr, eattr in nbrs.items():
                capacity = eattr['is_wave_avai']
                state_window = capacity[time:time + pl.WINDOW]
                if state_window.shape[0] < pl.WINDOW:
                    append_list = [[0 for _ in range(10)]]
                    for j in range(pl.WINDOW - state_window.shape[0]):
                        state_window = np.concatenate((state_window, append_list), axis=0)
                state[(10 * j):(10 * (j + 1))] = state_window
                j = j + 1

        # network state:request state
        holding_time = 0
        traffic = 0
        for base_index in range(self.max_iter):
            if self.request[base_index].arrival_time == time:
                holding_time = self.request[base_index].leave_time - time
                traffic = self.request[base_index].traffic
                break

        state_request = np.zeros(shape=(holding_time, 10))

        for m in range(holding_time):
            for n in range(traffic):
                state_request[m][n] = 1
        # print('request:', state_request)
        index = j * pl.WINDOW
        print('index: ', index)
        print(state.shape)
        state[index: (holding_time + index)] = state_request

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
        print('================a new action step==================')
        print('the time now is: ', self.time)
        print('the event now is: ', self.event_iter)
        print('+++++++++++++++++++++++++')
        if action is -1:
            return np.array([None, None]), 0, True, None
        #  --------------------------------------------------------------
        done = False
        info = False
        reward = 0
        # check if there are events (arrival of departure)
        if self.event[self.event_iter][0] > self.time:
            if action == self.k * self.wave_num:
                reward = NOARRIVAL_NO
            else:
                reward = NOARRIVAL_OT
            print('the next time is: ', self.time)
            observation = self.get_state_link(self.time)
            self.time += 1
        elif self.event[self.event_iter][0] == self.time:
            while self.event[self.event_iter][0] == self.time:
                if self.event[self.event_iter][2] is False:
                    req = self.request[self.event[self.event_iter][1]]
                    if hasattr(req, 'path'):
                        self.network.set_wave_state(wave_index=req.wave_index,
                                                    time_index=req.arrival_time,
                                                    holding_time=(req.leave_time-req.arrival_time+1),
                                                    nodes=req.path,
                                                    state=True,
                                                    check=True)
                    else:
                        pass
                    self.event_iter += 1
                else:
                    info = True
                    req = self.request[self.event[self.event_iter][1]]
                    reward = self.exec_action(action, req)
                    print('successfully mapped')
                    print('the time is: ', self.time)
                    print('the event index is: ', self.event_iter)
                    self.event_iter += 1

                if self.event_iter == len(self.event):
                    done = True
                    reward = 0
                    observation = self.get_state_link(self.time)
                    return observation, reward, done, info
            observation = self.get_state_link(self.time)
            self.time += 1
        else:
            raise EnvironmentError("there are some events not handled.")

        return observation, reward, done, info

    def exec_action(self, action: int, req: Request) -> float:
        """
        for the given request, execute the action and get the reward
        :param action:
        :param req:
        :return:
        """
        path_list = list(self.k_shortest_paths(req.src, req.dst))
        is_avai, _, _ = self.network.exist_rw_allocation(path_list, req.arrival_time, req.leave_time)
        print('the src is:', req.src)
        print('the dst is:', req.dst)
        print('the arrival time is: ', req.arrival_time)
        print('the leave time: ', req.leave_time)
        print('path_list:', path_list)
        print("is_avai:", is_avai)
        print('action is:', action)
        if action == self.NO_ACTION:
            if is_avai:
                return ARRIVAL_NO
            else:
                return ARRIVAL_NOOP
        else:
            if is_avai:
                route_index = action // (self.k * self.wave_num)
                wave_index = action % (self.k * self.wave_num)
                print('route_index: ', route_index)
                print('wave_index: ', wave_index)
                print('-------------------------------')
                if self.network.is_allocable(path_list[route_index], wave_index, req.arrival_time, req.leave_time):
                    self.network.set_wave_state(time_index=req.arrival_time,
                                                holding_time=(req.leave_time-req.arrival_time+1),
                                                wave_index=wave_index,
                                                nodes=path_list[route_index],
                                                state=False,
                                                check=True)
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
    game = Game(mode="LINN", total_time=20, wave_num=10, vm_num=10, max_iter=20, rou=2, mu=15, k=3, f=3, weight=1)
    # paths = list(game.k_shortest_paths(1, 6))
    # print(paths)
    # print(game.network.nodes[1]['capacity'])
    # print(game.network.edges[1, 3]['is_wave_avai'])
    game.reset()
    # path = [4, 1, 2, 5]
    # game.network.set_wave_state(time_index=5, holding_time=3, wave_index=1, nodes=path, state=False, check=True)
    # print(game.network.is_allocable(path=path, wave_index=1, start_time=5, end_time=7))

    print('--------------------the service requests---------------')
    # show_requests()
    print('--------------------the network state-------------------')
    done = False
    action = 1
    while not done:
        obs, reward, done, info = game.step(action)
        if done:
            print(done)
        # print('obs? ', obs, 'reward? ', reward, 'done? ', done, 'info? ', info)
    # game.network.show_link_state()
    # print(done)
    # request = Request(1, 1, 6, 1, 4, 1)
    # reward1 = game.exec_action(1, request)
    # print(reward1)
    pass
