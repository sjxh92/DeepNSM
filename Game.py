import numpy as np
import pandas as pd
from MetroNetwork import NetworkEnvironment as network
from Service import Service as service
import Service as Service


class Game(object):
    def __init__(self):
        self.network = network()
        self.service = service()
        self.action_space = []

    def action_space_gen(self, arrivalEvent):
        # transform the candidate paths into pandas format [path, weight]
        paths = Service.candidate_paths(arrivalEvent, 3)
        candidates = {}
        for path in paths:
            candidates['path'] = path
            candidates['weight'] = None
        paths_pandas = pd.DataFrame(candidates, None, columns=['path', 'weight'])
        # rank the candidate paths in the descending order of remaining capacity
        for path in paths_pandas:
            capacity = 0
            for link in path:
                capacity += link['capacity']
            path['weight'] += capacity
        paths_pandas.sort_values(by='weight', ascending=False)
        return paths_pandas
    def state_space_gen(self):
        pass

    def step(self):
        pass
