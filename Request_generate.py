import numpy as np
import random
import math

np.random.seed(0)


class RequestGenerate(object):

    def __init__(self, nodeNum):
        self.nodeNum = nodeNum

    def poissonNext(self, X, lamda):
        fx = 1 - math.exp(-lamda * X)
        r = random.uniform(0, 1)
        if r <= fx:
            traffic = np.random.uniform(25, 100)
            node = np.random.uniform(self.nodeNum)
            time = np.random.uniform(10, 50)
        else:
            traffic = -1
            node = -1
            time = -1
        return traffic, node, time
