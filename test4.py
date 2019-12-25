import numpy as np
import pandas as pd
import os
from test3 import person as p


class group(object):

    @staticmethod
    def show():
        candidates_pandas = pd.DataFrame(columns=('path', 'weight'))
        for i in range(6):
            candidates_pandas = candidates_pandas.append(
                pd.DataFrame({'path': [(1, 2, 3)], 'weight': [i]}), ignore_index=True)
            print(candidates_pandas)

        while True:
            print("fdsafdsafa")
            assert 1 == 2
            print("dsafdsafsa")


if __name__ == "__main__":
    # G = group()
    # G.show()
    file_prefix = "/home/mario/PycharmProjects/DeepNSM/Resource"
    filename = "7node_10link"
    filepath = os.path.join(file_prefix, filename)
    if os.path.isfile(filepath):
        datas = np.loadtxt(filepath, delimiter='|', skiprows=2, dtype=str)
        print(datas)
        origin_data = datas[:, 1:(datas.shape[1] - 1)]
        print(origin_data)
        print(origin_data.shape[0])
        print(origin_data.shape[1])
        for i in range(origin_data.shape[0]):
            wave_avai = [True for j in range(8)]
        print(wave_avai)

    else:
        raise FileExistsError("file {} doesn't exists.".format(filepath))
