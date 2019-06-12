from typing import Tuple, List, Callable

import numpy, random, math, pygame

from Nets.Net import Net
from formulas import distance_formula, sigmoid, sigmoid_der, color_formula


class MatrixNet(Net):
    def __init__(self,
                 dimensions: List[int],
                 weight_range: Tuple[float, float],
                 activation: Callable = sigmoid,
                 activation_der: Callable = sigmoid_der,
                 color_formula_param: Callable = color_formula):

        super(MatrixNet, self).__init__(dimensions[0], dimensions[-1], activation, activation_der, color_formula_param)
        self.input_array: numpy.array = numpy.array([[0]] * dimensions[0])
        self.weight_array: List[numpy.array] = []
        self.nodes_value_array: List[List[List[int]]] = []
        self.dimensions: List[int] = dimensions
        self.score: float = 0

        for i in range(1, len(dimensions)):
            node_array = []
            weight_array = []

            for ii in range(0, dimensions[i]):
                node_array.append([0])
                weight_array.append([])
                for iii in range(0, dimensions[i - 1] + 1):
                    weight_array[ii].append(random.random() * (weight_range[1] - weight_range[0]) + weight_range[0])

            self.weight_array.append(numpy.array(weight_array))
            self.nodes_value_array.append(numpy.array(node_array))

    def set_in(self, array: List[int]):
        if len(array) == len(self.input_array):
            for i in range(0, len(array)):
                if array[i] is not None:
                    self.input_array[i][0] = array[i]

    def get_out(self):
        self.nodes_value_array[0] = self.activation_function(
            self.weight_array[0].dot(
                numpy.reshape(numpy.append(self.input_array, 1.0), ((len(self.input_array) + 1), 1))))

        for i in range(1, len(self.nodes_value_array)):
            self.nodes_value_array[i] = self.activation_function(self.weight_array[i].dot(
                numpy.reshape(numpy.append(self.nodes_value_array[i - 1], 1.0),
                              ((len(self.nodes_value_array[i - 1]) + 1), 1))))
        return self.nodes_value_array[-1]

    def learn(self, ratio: float, target: List[int]):
        target_length = len(target)

        target = numpy.reshape(numpy.array([target]), (target_length, 1))

        past = numpy.multiply(2.0, (numpy.subtract(target, self.nodes_value_array[-1])))

        error = distance_formula(target, self.nodes_value_array[-1])

        for i in range(len(self.nodes_value_array) - 1, 0, -1):
            nodes_value_array_temp = self.nodes_value_array[i]

            nodes_value_array_temp2 = numpy.reshape(numpy.append(self.nodes_value_array[i - 1], 1),
                                                    (1, len(self.nodes_value_array[i - 1]) + 1))

            sigmoid_derivative = self.activation_function_derivative(nodes_value_array_temp)
            sigmoid_derivative_with_past = numpy.multiply(sigmoid_derivative, past)
            current = sigmoid_derivative_with_past.dot(nodes_value_array_temp2)
            past = numpy.transpose(sigmoid_derivative_with_past).dot(self.weight_array[i])
            past = numpy.reshape(past, (len(past[0]), 1))[:-1]
            current = numpy.multiply(current, ratio)
            self.weight_array[i] = numpy.add(self.weight_array[i], current)

        nodes_value_array_temp = self.nodes_value_array[0]

        nodes_value_array_temp2 = numpy.reshape(numpy.append(self.input_array, 1),
                                                (1, len(self.input_array) + 1))
        sigmoid_derivative = self.activation_function_derivative(nodes_value_array_temp)
        sigmoid_derivative_with_past = numpy.multiply(sigmoid_derivative, past)

        current = sigmoid_derivative_with_past.dot(nodes_value_array_temp2)
        current = numpy.multiply(current, ratio)
        self.weight_array[0] = numpy.add(self.weight_array[0], current)

        return error

    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, scale_dot: int = 5):
        scale_y = (height - scale_dot * 2) // max(self.dimensions)
        scale_x = (width - scale_dot * 2) // (len(self.dimensions) - 1)
        for y_ in range(0, len(self.input_array)):
            pygame.draw.circle(screen, self.color_formula(self.input_array[y_]),
                               [int(x + scale_dot), int(y + scale_dot + y_ * scale_y)],
                               int(scale_dot))
        for x_ in range(0, len(self.nodes_value_array)):
            for y_ in range(0, len(self.nodes_value_array[x_])):
                pygame.draw.circle(screen, self.color_formula(self.nodes_value_array[x_][y_]),
                                   [int(x + scale_dot + (x_ + 1) * scale_x), int(y + scale_dot + y_ * scale_y)],
                                   int(scale_dot))
                for y2 in range(0, len(self.weight_array[x_][y_])):
                    pygame.draw.line(screen,
                                     [255. - 255. * self.activation_function(self.weight_array[x_][y_][y2]), 125
                                         , 255. * self.activation_function(self.weight_array[x_][y_][y2])],
                                     [x + scale_dot + (x_ + 1) * scale_x, y + scale_dot + y_ * scale_y],
                                     [x + scale_dot + (x_) * scale_x, y + scale_dot + y2 * scale_y])
