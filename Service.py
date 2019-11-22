import pl
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from itertools import islice
from MetroNetwork import NetworkEnvironment as network

random.seed(1)


class Service(object):

    def __init__(self, lamda):
        self.lam = lamda
        self.N = pl.REQUEST_NUM
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

    def event_gen(self, n):
        i = 0
        while i < n:
            a, b = self.exponential_rand()
            traffic = random.randint(0, 10)
            source = random.randint(0, 6)
            self.arrivalEvents[i] = [i, int(a), traffic, source]
            departuretime = int(i + a)
            if departuretime < 1000:
                self.departureEvents[i] = [departuretime, i]
            else:
                self.departureEvents[i] = [1000, i]
            i += 1

    def k_shortest_paths(self, source, target, k, weight=None):
        return list(islice(nx.shortest_simple_paths(self, source, target, weight=weight), k))

    def candidate_paths(self, arrivalEvent, k):
        net = network()
        topology = net
        source = arrivalEvent[3]
        target = topology.nodes[-1]
        # traffic = arrivalEvent[2]
        paths = self.k_shortest_paths(source, target, k)
        return paths


if __name__ == "__main__":
    s = Service(0.1, 1000)
    s.event_gen()
    print(s.arrivalEvents)
    print(s.departureEvents)
