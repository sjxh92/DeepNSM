import numpy as np
from test3 import person as p


class group(object):
    def __init__(self):
        self.person = p('mario', '22', '173')

    def show(self):
        print(self.person.profile)


if __name__ == "__main__":
    G = group()
    G.show()
