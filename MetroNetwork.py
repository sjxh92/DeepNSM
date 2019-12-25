import numpy as np
import pandas as pd
import networkx as nx
import os
from itertools import islice
import random
import matplotlib.pyplot as plt

NODE_NUM = 7
LINK_NUM = 12
LAMDA = 3 / 10
J_NODE = 3
M_VM = 3
K_LINK = 3
W_WAVELENGTH = 3

COMPUTING_REQUIREMENT = 5
BANDWIDTH_REQUIREMENT = 2


class NetworkEnvironment(nx.Graph):

    def __init__(self, filename: str, file_prefix: str, wave_num: int, total_step: int, vm_num: int):
        # self.n_feature = NODE_NUM + 2 + J_NODE * M_VM + K_LINK * W_WAVELENGTH

        # node utilization + link utilization + request node + request traffic + holding time
        super(NetworkEnvironment, self).__init__()
        self.net = None
        self.n_feature = NODE_NUM + LINK_NUM + 1 + 1
        self.action_space = []
        self.wave_num = wave_num
        self.total_step = total_step
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
                self.add_node(i + 1, capacity=np.zeros(shape=(total_step, vm_num), dtype=int))  # time step
            for i in range(self.origin_data.shape[0]):
                wave_avai = np.zeros(shape=(total_step, wave_num), dtype=bool)
                for j in range(total_step):
                    wave_avai[j] = [True for k in range(wave_num)]
                self.add_edge(int(self.origin_data[i, 1]), int(self.origin_data[i, 2]),
                              weight=float(self.origin_data[i, 3]), is_wave_avai=wave_avai)
        else:
            raise FileExistsError("file {} doesn't exists.".format(filepath))

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

    def set_wave_state(self, wave_index, nodes: list, state: bool, check: bool = True):
        """
        set the state of a certain wavelength on a path
        :param wave_index:
        :param nodes:
        :param state:
        :param check:
        :return:
        """
        assert len(nodes) >= 2
        start_node = nodes[0]
        for i in range(1, len(nodes)):
            end_node = nodes[i]
            if check:
                assert self.get_edge_data(start_node, end_node)['is_wave_avai'][wave_index] != state
            self.get_edge_data(start_node, end_node)['is_wave_avai'][wave_index] = state
            start_node = end_node

    def exist_rw_allocation(self, path_list: list) -> [bool, int, int]:
        """
        check all the paths and all the wavelengths in path list
        :param path_list:
        :return:
        """
        if len(path_list) == 0 or path_list[0] is None:
            return False, -1, -1

        for path_index, nodes in enumerate(path_list):
            edges = self.extract_path(nodes)
            for wave_index in range(self.wave_num):
                is_avai = True
                for edge in edges:
                    if self.get_edge_data(edge[0], edge[1])['is_wave_avai'][wave_index] is False:
                        is_avai = False
                        break
                if is_avai is True:
                    return True, path_index, wave_index

    def is_allocable(self, path: list, wave_index: int) -> bool:
        """
        if the wave_index in path is available
        :param path:
        :param wave_index:
        :return:
        """
        print('wave_index')
        print(wave_index)
        edges = self.extract_path(path)
        is_avai = True
        for edge in edges:
            print(self.get_edge_data(edge[0], edge[1])['is_wave_avai'])
            if self.get_edge_data(edge[0], edge[1])['is_wave_avai'][wave_index] is False:
                is_avai = False
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


if __name__ == "__main__":
    TP = NetworkEnvironment("7node_10link", "/home/mario/PycharmProjects/DeepNSM/Resource", 8, 10, 10)
    print(TP.nodes.data())
    # print(TP.edges.data())
