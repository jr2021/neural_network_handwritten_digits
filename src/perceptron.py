import math
import os
from scipy.spatial import distance
import numpy as np
import random

def get_characters(filename):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    chars = []
    with open(os.path.join(dir_path, '..', filename)) as file:
        for line in file:
            chars.append(line[0])
    return chars


def get_images(filename):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    vectors = []
    with open(os.path.join(dir_path, '..', filename)) as file:
        for line in file:
            vectors.append([1.0 if float(v) == 1 else -1.0 for v in line.strip().split(',')])
    return vectors


class Perceptron:

    def __init__(self, image_file, class_file):
        vectors = get_images(image_file)
        classes = get_characters(class_file)
        self.data = [{'x': v, 'y': c} for (v, c) in zip(vectors, classes)]
        random.seed()
        self.w = np.array([0] * 784)

    def train(self, digit, not_digit, steps):
        for datum in self.data[:5000]:
            if datum["y"] in (digit, not_digit):
                z = np.dot(datum["x"], self.w)
                if z >= 0 and datum["y"] != digit:
                    self.w = np.subtract(self.w, datum["x"])
                if z < 0 and datum["y"] == digit:
                    self.w = np.add(self.w, datum["x"])

    def test(self, digit, not_digit):
        success = 0
        failure = 0
        total = 0
        for datum in self.data[5000:]:
            if datum["y"] in (digit, not_digit):
                z = np.dot(datum['x'], self.w)
                if z >= 0 and datum['y'] == digit:
                    success += 1
                elif z < 0 and datum['y'] == not_digit:
                    success += 1
                else:
                    failure += 1
                total += 1
        return float(success) / total
