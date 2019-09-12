import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from itertools import islice
from MetroNetwork import NetworkEnvironment as network

random.seed(1)


def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def candidate_paths(arrivalEvent, k):
    net = network()
    topology = net.topology
    source = arrivalEvent[3]
    target = topology.nodes[-1]
    # traffic = arrivalEvent[2]
    paths = k_shortest_paths(topology, source, target, k)
    return paths


class Service(object):

    def __init__(self, lamda, N):
        self.lam = lamda
        self.N = N
        self.arrivalEvents = np.empty(shape=(1000, 4))  # (starting time, holding time, traffic demand, source node)
        self.departureEvents = np.zeros(shape=(1000, 2))  # (departure time, connection id)

        self.connections = pd.DataFrame({'id': int,
                                         'path': list,
                                         'wavelength': list,
                                         'traffic': int,
                                         'ta': int,
                                         'th': int})
        self.net = network()

    def exponential_pdf(self, x):
        return self.lam * math.exp(-self.lam * x)

    def exponential_rand(self):
        while True:
            a = random.uniform(0.0, 1000.0)
            b = random.uniform(0.0, 50.0)
            if b <= self.exponential_pdf(a):
                return a, b

    def event_gen(self):
        i = 0
        while i < s.N:
            a, b = self.exponential_rand()
            traffic = random.randint(0, 10)
            source = random.randint(0, 6)
            self.arrivalEvents[i] = [i, int(a), traffic, source]
            departuretime = int(i + a)
            if departuretime < 1000:
                self.departureEvents[i] = [departuretime, i]
            i += 1

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


if __name__ == "__main__":
    s = Service(0.1, 1000)
    s.event_gen()
    print(s.arrivalEvents)
    print(s.departureEvents)
