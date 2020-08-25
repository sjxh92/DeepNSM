import numpy as np
import pandas as pd
import networkx as nx
import os
from itertools import islice
import random
import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelno)s - %(name)s - '
                                                '%(levelname)s - %(filename)s - %(funcName)s - '
                                                '%(message)s')
logger = logging.getLogger(__name__)

NODE_NUM = 7
LINK_NUM = 20


class NetworkEnvironment(nx.Graph):

    def __init__(self, filename: str, file_prefix: str, wave_num: int, vm_num: int, total_time: int):
        # self.n_feature = NODE_NUM + 2 + J_NODE * M_VM + K_LINK * W_WAVELENGTH

        # node utilization + link utilization + request node + request traffic + holding time
        super(NetworkEnvironment, self).__init__()
        self.net = None
        self.action_space = []
        self.wave_num = wave_num
        self.total_time = total_time
        self.vm_num = vm_num
        self.n_action = len(self.action_space)
        self.X = 0
        self.memory = np.zeros([0, 4], dtype=int)  # nodeid, traffic, starttime, endtime
        # self._topology()

        filepath = os.path.join(file_prefix, filename)
        if os.path.isfile(filepath):
            datas = np.loadtxt(filepath, delimiter='|', skiprows=2, dtype=str)
            self.origin_data = datas[:, 1:(datas.shape[1] - 1)]
            print(type(self.origin_data[1, 1]))
            for i in range(NODE_NUM):
                self.add_node(i + 1, capacity=np.zeros(shape=(total_time, vm_num), dtype=int))  # time step
            for i in range(self.origin_data.shape[0]):
                wave_avai = np.zeros(shape=(total_time, wave_num), dtype=bool)
                for j in range(total_time):
                    wave_avai[j] = [True for k in range(wave_num)]
                self.add_edge(int(self.origin_data[i, 1]), int(self.origin_data[i, 2]),
                              weight=float(self.origin_data[i, 3]), is_wave_avai=wave_avai)
        else:
            raise FileExistsError("file {} doesn't exists.".format(filepath))

    def reset(self):
        for i in range(NODE_NUM):
            for j in range(self.nodes[i+1]['capacity'].shape[0]):
                for k in range(self.nodes[i+1]['capacity'].shape[1]):
                    self.nodes[i+1]['capacity'][j][k] = 0
        for u, v in self.edges:
            for m in range(self.get_edge_data(u, v)['is_wave_avai'].shape[0]):
                for n in range(self.get_edge_data(u, v)['is_wave_avai'].shape[1]):
                    self.get_edge_data(u, v)['is_wave_avai'][m][n] = True

    def set_wave_state(self, start_time: int,
                       end_time: int,
                       wave_index: int,
                       path: list,
                       state: bool,
                       check: bool = True):
        """
        set the state of a certain wavelength on a path
        :param path: node list of path
        :param start_time:
        :param end_time:
        :param wave_index:
        :param state:
        :param check:
        :return:
        """
        assert len(path) >= 2
        start_node = path[0]
        wave_index_sorted = self.wave_rank(path, start_time, end_time, wave_index)
        for i in range(1, len(path)):
            end_node = path[i]
            #  logger.info('the allocated link is: ', start_node, '-->', end_node, self.edges[start_node, end_node])
            for j in range(end_time - start_time + 1):
                if check:
                    assert self.get_edge_data(start_node, end_node)['is_wave_avai'][start_time+j][wave_index_sorted] != state
                self.get_edge_data(start_node, end_node)['is_wave_avai'][start_time+j][wave_index_sorted] = state
            start_node = end_node

    def set_node_state(self, start_time: int,
                       end_time: int,
                       path: list,
                       node_index: int,
                       demand: int,
                       state: int):
        """

        :param path:
        :param node_index:
        :param start_time:
        :param end_time:
        :param demand:
        :param state:
        :return:
        """
        node_index_sorted = self.node_rank(path, start_time, end_time, node_index)
        for i in range(end_time - start_time + 1):
            demand = demand // 10
            demand = demand + np.sum(self.nodes[node_index_sorted]['capacity'][start_time+i])
            for j in range(demand):
                self.nodes[node_index_sorted]['capacity'][start_time+i][j] = state
        return node_index_sorted

    def exist_rw_allocation(self, path_list: list, start_time: int, end_time: int) -> [bool, int, int]:
        """
        check all the paths and all the wavelengths in path list
        :param end_time:
        :param start_time:
        :param path_list:
        :return:
        """
        if len(path_list) == 0 or path_list[0] is None:
            return False, -1, -1

        if end_time > self.total_time - 1:
            return False, -1, -1

        for path_index, nodes in enumerate(path_list):
            edges = self.extract_path(nodes)
            for wave_index in range(self.wave_num):
                w_avai = True
                for edge in edges:
                    for time in range(end_time-start_time+1):
                        if self.get_edge_data(edge[0], edge[1])['is_wave_avai'][start_time+time][wave_index] is False:
                            w_avai = False
                            break
                    if w_avai is False:
                        break
                if w_avai is True:
                    return True, path_index, wave_index

        return False, -1, -1

    def wave_rank(self, path: list, start_time: int, end_time: int, wave_index: int):
        edges = self.extract_path(path)
        wave_weight = np.arange(self.wave_num)
        for wave in range(self.wave_num):
            wave_sum = 0
            for time in range(end_time - start_time + 1):
                for edge in edges:
                    wave_sum = wave_sum + self.get_edge_data(edge[0], edge[1])['is_wave_avai'][start_time + time][wave]
            wave_weight[wave] = wave_sum
        sorted_wave_weight = np.argsort(-wave_weight)
        wave_index_sorted = sorted_wave_weight[wave_index]
        return wave_index_sorted

    def node_rank(self, path: list, start_time: int, end_time: int, node_index: int):
        node_weight = np.arange(len(path))
        for node in range(len(path)):
            node_sum = 0
            for time in range(end_time - start_time + 1):
                node_sum = node_sum + np.sum(self.nodes[path[node]]['capacity'][start_time + time])
            node_weight[node] = node_sum
        sorted_node_weight = np.argsort(node_weight)
        node_index_sorted = sorted_node_weight[node_index]
        physical_node_index = path[node_index_sorted]
        return physical_node_index

    def is_allocable(self, req, path: list, wave_index: int, node_index: int, demand: int, start_time: int, end_time: int) -> bool:
        """
        if the wave_index in path is available
        :param req:
        :param demand:
        :param node_index:
        :param end_time:
        :param start_time:
        :param path:
        :param wave_index:
        :return:
        """

        if end_time >= self.total_time:
            return False

        edges = self.extract_path(path)

        wave_index_sorted = self.wave_rank(path, start_time, end_time, wave_index)
        physical_node_index = self.node_rank(path, start_time, end_time, node_index)

        is_avai = True
        for edge in edges:
            # print("the link:", edge[0], "-->", edge[1], "is: ", self.get_edge_data(edge[0], edge[1])['is_wave_avai'])
            for time in range(end_time-start_time+1):
                if not self.get_edge_data(edge[0], edge[1])['is_wave_avai'][start_time+time][wave_index_sorted]:
                    is_avai = False
                    print('\033[1;32;40m the bandwidth is not enough', req.index)
                    break
            if is_avai is False:
                break
        for time in range(end_time-start_time+1):
            if np.sum(self.nodes[physical_node_index]['capacity'][start_time+time]) > 10 - demand:
                is_avai = False
                print('\033[1;32;40m the processing is not enough', req.index)
                break
        return is_avai

    def extract_path(self, nodes: list) -> list:
        # print(" extract path ")
        # print(nodes)
        assert len(nodes) >= 2
        rtn = []
        start_node = nodes[0]
        for i in range(1, len(nodes)):
            end_node = nodes[i]
            rtn.append((start_node, end_node))
            start_node = end_node
        return rtn

    def show_link_state(self):
        for n, nbrs in self.adjacency():
            print(n, nbrs.items())
            for nbr, eattr in nbrs.items():
                data = eattr['weight']
                wave_state = eattr['is_wave_avai']
                print('(%d, %d, %d, %0.3f)' % (n, nbr, data, wave_state))


if __name__ == "__main__":
    TP = NetworkEnvironment("7node_10link", "/home/mario/PycharmProjects/DeepNSM/Resource", 8, 10, 10)
    print(TP.nodes.data())
    print(TP.edges.data())
    for s, d in TP.edges:
        print(TP.get_edge_data(s, d))
    TP.reset()
    print('------------------------------')
    for s, d in TP.edges:
        print(TP.get_edge_data(s, d))
    print(TP.get_edge_data(5, 7)['is_wave_avai'])

