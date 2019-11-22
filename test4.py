import numpy as np
import pandas as pd
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
    G = group()
    G.show()
