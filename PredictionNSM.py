import numpy as np
import matplotlib.pyplot as plt
import TrafficModel
import MetroNetwork
import networkx as nx


class PredictionNSM(object):
    def __init__(self, network: MetroNetwork, n: int, total_time: int):
        self.network = network
        traffic_model = TrafficModel.SliceGenerator(n, network, total_time)
        self.slices = traffic_model.randomGenerate()
        self.time = 0
        self.transfer_traffic = []

    def initialize(self):
        for s in self.slices:
            src = s.src
            dst = s.dst
            paths = self.ksp(src, dst, 3)
            for path in paths:
                start_node = path[0]
                for i in range(1, len(path)):
                    end_node = path[i]
                    link = self.network.get_edge_data(start_node, end_node)[0][]
        pass

    def sliceAdjust(self):
        pass

    def bandwidthPrediction(self, time):
        pass

    def capacityPrediction(self, time):
        pass

    def findNode(self, time):
        pass

    def findPath(self, time):
        pass

    def checkPath(self, src, dst, time, traffic):
        paths = self.ksp(src, dst, 3)
        for path in paths:
            start_node = path[0]
            for i in range(len(path)):
                for j in range(10):
                    

    def ksp(self, source, target, k):
        """
        calculate the paths
        :param k:
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
            if index > k:
                break
            path_k.append(i)
        return path_k

    def showRequest(self):
        for s in self.slices:
            print('|slice id: ', s.index, '|slice src: ', s.src,
                  '|slice dst: ', s.dst, '|slice traffic', s.traffic,
                  '|total time: ', s.t_time)

    def showResults(self):
        pass


if __name__ == "__main__":
    heuristic = PredictionNSM(1, 3, 10)
    print(heuristic.slices)
    heuristic.showRequest()
