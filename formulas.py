import random

import math
import numpy


def distance_formula(a, b):
    return numpy.linalg.norm(numpy.subtract(a, b))


def sigmoid(x):
    return numpy.divide(1.0, numpy.add(1.0, numpy.exp(numpy.negative(x))))


def sigmoid_der(array):
    return numpy.multiply(numpy.subtract(1.0, array), array)


def tanh_derivative(x):
    return numpy.subtract(1.0, numpy.square(x))


tanh = numpy.tanh


def rand(num):
    return random.random() + num


randomize = numpy.vectorize(rand)
