import numpy as np
import MetroNetwork

np.random.seed(0)


class Request(object):
    def __init__(self, index: int, src: int, dst: int, total_time: int, traffic: list):
        super(Request, self).__init__()
        self.index = index
        self.src = src
        self.dst = dst
        self.t_time = total_time
        self.traffic = traffic
        self.wave_list = []
        self.node_list = []
        self.path_list = []

    def add_allocation(self, path: list, node: int, wave: int):
        self.path_list.append(path)
        self.node_list.append(node)
        self.wave_list.append(wave)


class SliceGenerator(object):
    def __init__(self, n: int, network: MetroNetwork, total_time: int):
        self.n = n
        self.network = network
        self.total_time = total_time
        self.slices = []
        pass

    def randomGenerate(self):
        for i in range(self.n):
            index = i
            src = np.random.randint(0, 7)
            dst = 7
            traffic = []
            for t in range(self.total_time):
                if t // 5 == 0:
                    traffic_t = np.random.randint(1, 3)
                    traffic.append(traffic_t)
                elif t // 5 == 1:
                    traffic_t = np.random.randint(3, 7)
                    traffic.append(traffic_t)
                elif t // 5 == 2:
                    traffic_t = np.random.randint(7, 10)
                    traffic.append(traffic_t)
                elif t // 5 == 3:
                    traffic_t = np.random.randint(5, 7)
                    traffic.append(traffic_t)
                else:
                    traffic_t = np.random.randint(3, 5)
                    traffic.append(traffic_t)
            # print(traffic)
            slice_i = Request(index=index, src=src, dst=dst,
                              total_time=self.total_time, traffic=traffic)
            self.slices.append(slice_i)
        return self.slices

    def predictionGenerate(self):
        pass
