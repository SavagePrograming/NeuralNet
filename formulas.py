import random

import math
import numpy
import pygame


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


def color_formula(x):
    return [0, int(x * 255.), 0]


def map_helper(formula, *args):
    return list(map(formula, *args))


def map_helper_clean(formula, *args):
    return any(map(formula, *args))


def color_formula_line(x):
    return [255. - 255. * sigmoid(x), 125, 255. * sigmoid(x)]


def color_formula_line_helper(x):
    return list(map(color_formula_line, x))


def draw_circle(screen_range, color_range, in_range_loc, radius_range):
    pygame.draw.circle(screen_range, color_range, in_range_loc, radius_range)


def draw_circle_helper(screen_range, color_range, in_range_loc, radius_range):
    any(map(draw_circle, screen_range, color_range, in_range_loc, radius_range))


def draw_line_enable(screen, color, start_pos, end_pos, width, enable):
    # print("=============================")
    # print(screen)
    # print(color)
    # print(start_pos)
    # print(end_pos)
    # print(width)
    # print(enable)
    if enable:
        pygame.draw.line(screen, color, start_pos, end_pos, width)


def draw_line_helper(screen, color, start_pos, end_pos, width, enable):
    # print("++++++++++++++++++++++++")
    # print(dim(screen))
    # print(dim(color))
    # print(dim(start_pos))
    # print(dim(end_pos))
    # print(dim(width))
    # print(dim(enable))
    any(map(draw_line_enable, screen, color, start_pos, end_pos, width, enable))


def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])
