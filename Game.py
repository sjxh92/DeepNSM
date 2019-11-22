import numpy as np
import pandas as pd
from MetroNetwork import NetworkEnvironment as network
from Service import Service as service
import Service
from args import args
import pl
import random
import chainer
import chainerrl

# there is no request arriving or leaving at this time
NOARRIVAL_NO = 1  # choose no_action
NOARRIVAL_OT = -1  # choose other actions

INIT = 0


class Request(object):
    def __init__(self, index: int, src: str, dst: str, arrival_time: int, leave_time: int, traffic: int):
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
    def __init__(self, mode: str, max_iter: int, rou: float, mu: float, k: int, f: int):
        """

        :param mode:
        :param max_iter: max number of requests
        :param rou:
        :param mu:
        :param k: ksp
        :param f: f candidate node for du/cu placement
        """
        self.network = network()
        self.action_space = []
        self.mode = mode
        self.max_iter = max_iter
        self.rou = rou
        self.mu = mu
        self.k = k
        self.f = f

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

    # generate (network state and request state)
    '''
    |resource|
    (*****,——
    *****,  t
    *****,  i
    ...     m
    *****)  e
           --
    '''

    def state_space_gen(self, t, holdingtime, traffic):

        state = np.zeros(shape=(300, 10))

        # network state:node state
        for i in range(pl.NODE_NUM):
            node = self.network.nodes[i]
            capacity = node['utilization']
            state_window = capacity[t:t + pl.WINDOW]
            state[(10 * i):(10 * (i + 1))] = state_window
            # print(capacity)

        # network state:link state
        j = pl.NODE_NUM
        for n, nbrs in self.network.adjacency():
            for nbr, eattr in nbrs.items():
                capacity = eattr['utilization']
                state_window = capacity[t:t + pl.WINDOW]
                state[(10 * j):(10 * (j + 1))] = state_window
                j = j + 1

        # network state:request state

        state_request = np.zeros(shape=(10, 10))
        for m in range(holdingtime):
            for n in range(traffic):
                state_request[m][n] = 1
        state[(10 * j):(10 * (j + 1))] = state_request

        return state
        # print(state[100:150])
        pass

    def reset(self):

        """
        reset event and environment
        :return:
        """

        self.event_iter = 0
        self.event = []
        self.network = network()
        self.time = 0

        base_time = 0
        rand_val = int(random.random() * 1000000000)
        np.random.seed(rand_val)
        for base_index in range(self.max_iter):
            src = random.randint(0, pl.NODE_NUM)
            arrival = np.random.poisson(lam=self.rou) + base_time + 1
            leave = np.random.poisson(lam=self.mu) + base_time + 1
            traffic = random.randint(0, 10)
            self.request[base_index] = Request(base_index, src, '', arrival, leave, traffic)
            self.event.append([arrival, base_index, True])
            self.event.append([leave, base_index, False])

            base_time = arrival
        self.event.sort(key=lambda time: time[0])

        # return the first state
        arr, lea, tra = self.request[0].arrival, self.request[0].leave, self.request[0].traffic
        observation = self.state_space_gen(0, lea - arr, tra)
        reward = INIT
        done = False
        info = None
        self.time = self.request[0].arrival_time
        return observation, reward, done, info

    def get_state_link(self, time):

        state = np.zeros(shape=(300, 10))

        # network state:node state
        for i in range(pl.NODE_NUM):
            node = self.network.nodes[i]
            capacity = node['utilization']
            state_window = capacity[time:time + pl.WINDOW]
            state[(pl.WINDOW * i):(pl.WINDOW * (i + 1))] = state_window
            # print(capacity)

        # network state:link state
        j = pl.NODE_NUM
        for n, nbrs in self.network.adjacency():
            for nbr, eattr in nbrs.items():
                capacity = eattr['utilization']
                state_window = capacity[time:time + pl.WINDOW]
                state[(10 * j):(10 * (j + 1))] = state_window
                j = j + 1

        # network state:request state
        state_request = np.zeros(shape=(10, 10))

        holdingtime = 0
        traffic = 0
        for base_index in range(self.max_iter):
            if self.request[base_index].arrival_time == time:
                holdingtime = self.request[base_index].leave_time - time
                traffic = self.request[base_index].traffic

        for m in range(holdingtime):
            for n in range(traffic):
                state_request[m][n] = 1

        state[(10 * j):(10 * (j + 1))] = state_request

        return state
        # print(state[100:150])
        pass

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

        done = False
        info = False
        # check if there are events (arrival of departure)
        if self.event[self.event_iter][0] > self.time:
            # if there is no event at this time, any action make no sense
            if action == self.k * self.f:
                # block
                reward = NOARRIVAL_NO
            else:
                # choose other action
                reward = NOARRIVAL_OT
            self.time += 1
        elif self.event[self.event_iter][0] == self.time:
            # there is request arriving or leaving at this time
            # if there is a leaving request
            if self.event[self.event_iter][2] is False:
                raise RuntimeError("somthing wrong, there should not be a leaving request!!!")
            else:
                # if it is a arrving event, process it according to the action
                info = True
                req = self.request[self.event[self.event_iter][1]]
                reward = self.exec_action(action, req)

                if self.event[self.event_iter][1] == (self.max_iter - 1):
                    observation = self.state_space_gen(self.time, 0, 0)
                    done = True
                    return observation, reward, done, info

                self.event_iter += 1
                while self.event[self.event_iter][0] == self.time:
                    # handle the departure event
                    assert self.event[self.event_iter][2] is False
                    leave_request = self.service[self.event[self.event_iter][1]]
                    if hasattr(leave_request, 'path'):
                        self.network.set_wave_state(wave_index=leave_request.wave_index,
                                                    nodes=leave_request.path,
                                                    state=True,
                                                    check=True)
                    else:
                        pass
                    self.event_iter += 1
                self.time += 1
        else:
            raise EnvironmentError("there are some events remaining to be handled")

        # check if the time is near ending
        if self.event_iter == len(self.event):
            # if all the events have been handled
            done = True
            observation = self.state_space_gen(self.time, 0, 0)
            return observation, reward, done, info

        # handle the next state
        if self.event[self.event_iter][0] > self.time:

            observation = None
        elif self.event[self.event_iter][0] == self.time:
            # handle the former departure event
            while self.event[self.event_iter][2] is False and self.event[self.event_iter][0] == self.time:
                leave_request = self.request[self.event[self.event_iter][1]]
                if hasattr(leave_request, 'path'):
                    self.network.set_wave_state(wave_index=leave_request.wave_index,
                                                nodes=leave_request.path,
                                                state=True,
                                                check=True)
                else:
                    pass
                self.event_iter += 1
                if self.event_iter == len(self.event):
                    done = True
                    observation = self.state_space_gen(self.time, 0, 0)
                    return observation, reward, done, info

            if self.event[self.event_iter][0] == self.time:
                assert self.event[self.event_iter][2] is True
                request = self.request[self.event[self.event_iter][1]]
                observation = self.get_state_link(self.time)
            else:
                observation = self.get_state_link(self.time)
        else:

            raise EnvironmentError("some events are not handled")
        return observation, reward, done, info

    def exec_action(self, action: int, request: Request) -> float:
        """
        for the given request, execute the action and get the reward
        :param action:
        :param request:
        :return:
        """
        path_list = self

        return None
        pass


if __name__ == "__main__":
    game = Game()
    game.state_space_gen(3)
    pass
