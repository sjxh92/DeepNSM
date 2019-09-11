import numpy as np


class person(object):
    profile = None

    def __init__(self, name, age, height):
        self.name = name
        self.age = age
        self.height = height
        self.profile = []
        self.generate()

    def generate(self):
        self.profile.append(self.name)
        self.profile.append(self.age)
        self.profile.append(self.height)
