import numpy

distance_formula = numpy.linalg.norm


def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))