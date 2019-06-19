from typing import List, Callable, Tuple

import numpy, math, pygame

from Nets.Net import Net
from formulas import distance_formula, sigmoid, sigmoid_der, randomize, color_formula, color_formula_line, \
    color_formula_line_helper, draw_circle, draw_line_helper, encode_list, decode_list, to_bool


class LinearNet(Net):
    def __init__(self,
                 in_dem: int,
                 out_dem: int,
                 middle_dem: int,
                 weight_range: Tuple[float, float] = [2.0, -2.0],
                 enabled_weights: numpy.array = None,
                 activation: Callable = sigmoid,
                 activation_der: Callable = sigmoid_der,
                 color_formula_param: Callable = color_formula,
                 weights: numpy.array = None):
        super(LinearNet, self).__init__(in_dem + 1, out_dem, activation, activation_der, color_formula_param)
        self.middle_dem: int = middle_dem

        self.score = 0
        self.input_nodes = numpy.zeros((1, self.in_dem))

        self.node_values = numpy.zeros((1, self.middle_dem + self.out_dem))
        self.node_sum = numpy.zeros((1, self.middle_dem + self.out_dem))
        self.node_back = numpy.zeros((1, self.in_dem + self.middle_dem))
        if weights is not None:
            self.weights = weights
        else:
            dif = abs(weight_range[0] - weight_range[1])
            self.weights = randomize(numpy.zeros((self.in_dem + self.middle_dem, self.middle_dem + self.out_dem)))
            self.weights = numpy.add(weight_range[1], numpy.multiply(dif, self.weights))

        if enabled_weights:
            self.enabled_weights: numpy.array = enabled_weights
        else:
            self.enabled_weights: numpy.array = numpy.array([
                [in_node < out_node
                 for out_node in range(self.in_dem, self.in_dem + self.middle_dem + self.out_dem)]
                for in_node in range(self.in_dem + self.middle_dem)])

    def update(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, scale_dot: int = 5):

        in_spacing = (height - scale_dot * 2) / (self.in_dem + 1)
        out_spacing = (height - scale_dot * 2) / (self.out_dem + 1)

        center_x = x + width / 2  # x + width / 4
        center_y = y + height - scale_dot
        radius_y = height - scale_dot * 2
        radius_x = width / 4  # width / 2
        angle = 0
        if self.middle_dem > 1:
            angle = math.pi / (self.middle_dem - 1)

        self.in_screen_range = [screen] * self.in_dem
        self.middle_screen_range = [screen] * self.middle_dem
        self.out_screen_range = [screen] * self.out_dem
        self.line_screen_range = [[screen] * (self.out_dem + self.middle_dem)] * (self.middle_dem + self.in_dem)

        self.in_range_loc = numpy.zeros((self.in_dem, 2)).astype(int)
        self.in_range_loc[:, 0:1] = (numpy.add(x + scale_dot, self.in_range_loc[:, 0:1]))
        self.in_range_loc[:, 1:2] = (numpy.add(y + scale_dot, numpy.multiply(
            numpy.add(numpy.reshape(range(self.in_dem), (self.in_dem, 1)), 1), in_spacing)))

        self.middle_range_loc = numpy.concatenate([
            numpy.reshape(numpy.subtract(center_x, numpy.multiply(radius_x, numpy.cos(
                numpy.multiply(angle, range(self.middle_dem))))),
                          (self.middle_dem, 1)),
            numpy.reshape(numpy.subtract(center_y, numpy.multiply(radius_y, numpy.sin(
                numpy.multiply(angle, range(self.middle_dem))))),
                          (self.middle_dem, 1)),
        ], 1).astype(int)

        self.out_range_loc = numpy.zeros((self.out_dem, 2)).astype(int)
        self.out_range_loc[:, 0:1] = (numpy.add(x + width - scale_dot, self.out_range_loc[:, 0:1]))
        self.out_range_loc[:, 1:2] = (numpy.add(y + scale_dot, numpy.multiply(
            numpy.add(numpy.reshape(range(self.out_dem), (self.out_dem, 1)), 1), out_spacing)))

        self.in_radius_range = [scale_dot] * self.in_dem
        self.middle_radius_range = [scale_dot] * self.middle_dem
        self.out_radius_range = [scale_dot] * self.out_dem
        self.line_radius_range = [[1] * (self.out_dem + self.middle_dem)] * (self.middle_dem + self.in_dem)

        # self.line_start_loc = [numpy.concatenate([self.in_range_loc, self.middle_range_loc], 0)] * (
        #         self.middle_dem + self.out_dem)
        # self.line_end_loc = [[end] * (self.in_dem + self.middle_dem) for end in
        #                      numpy.concatenate([self.middle_range_loc, self.out_range_loc], 0)]

        self.line_start_loc = [[start] * (self.out_dem + self.middle_dem) for start in
                               numpy.concatenate([self.in_range_loc, self.middle_range_loc], 0)]
        self.line_end_loc = [numpy.concatenate([self.middle_range_loc, self.out_range_loc], 0)] * (
                self.middle_dem + self.in_dem)

        self.update_colors()

    def update_colors(self):
        self.in_color_range = list(map(self.color_formula, self.input_nodes[0]))
        self.middle_color_range = list(map(self.color_formula, self.node_values[0][:self.middle_dem]))
        self.out_color_range = list(map(self.color_formula, self.node_values[0][self.middle_dem:]))
        self.line_color_range = list(map(color_formula_line_helper, self.weights))

    def draw(self):
        self.update_colors()

        list(map(draw_line_helper, self.line_screen_range, self.line_color_range,
                 self.line_start_loc, self.line_end_loc, self.line_radius_range, self.enabled_weights))

        any(map(draw_circle, self.in_screen_range, self.in_color_range, self.in_range_loc, self.in_radius_range))

        any(map(draw_circle, self.middle_screen_range, self.middle_color_range, self.middle_range_loc,
                self.middle_radius_range))

        any(map(draw_circle, self.out_screen_range, self.out_color_range, self.out_range_loc,
                self.out_radius_range))

    def set_in(self, array: List[int]):
        array = array + [1.0]
        assert len(array) == self.in_dem
        self.input_nodes = numpy.array(array, ndmin=2)

    def get_out(self):

        self.weights = numpy.multiply(self.weights, self.enabled_weights)
        numpy.zeros((1, self.middle_dem + self.out_dem))
        self.node_sum = numpy.dot(self.input_nodes, self.weights[:self.in_dem])

        for i in range(self.middle_dem):
            self.node_values[0][i] = self.activation_function(self.node_sum[0][i])

            self.node_sum = numpy.add(self.node_sum, numpy.multiply(self.node_values[0][i],
                                                                    self.weights[self.in_dem + i:self.in_dem + i + 1]))

        self.node_values = self.activation_function(self.node_sum)

        return self.node_values[0][self.middle_dem:]

    def learn(self, ratio: int, target: List[int]):
        self.weights = numpy.multiply(self.weights, self.enabled_weights)
        target = numpy.reshape(target, (1, len(target)))

        difference = numpy.multiply(2.0, numpy.subtract(target, self.node_values[:, self.middle_dem:]))

        past = numpy.multiply(self.activation_derivative(self.node_values[:, self.middle_dem:]), difference)

        self.node_back = numpy.dot(past, numpy.transpose(self.weights[:, self.middle_dem:]))

        weight_shift = numpy.multiply(ratio, numpy.dot(
            numpy.transpose(numpy.concatenate((self.input_nodes, self.node_values[:, :self.middle_dem]), axis=1)),
            past))

        self.weights[:, self.middle_dem:] = numpy.add(weight_shift, self.weights[:, self.middle_dem:])

        for n in range(self.middle_dem - 1, -1, -1):
            past = numpy.multiply(self.activation_derivative(self.node_values[0][n]),
                                  self.node_back[0][self.in_dem + n])

            self.node_back = numpy.add(numpy.multiply(past, numpy.transpose(self.weights[:, n])), self.node_back)

            weight_shift = numpy.multiply(ratio, numpy.multiply(
                numpy.transpose(numpy.concatenate((self.input_nodes, self.node_values[:, :self.middle_dem]), axis=1)),
                past))

            self.weights[:, n:n + 1] = numpy.add(weight_shift, self.weights[:, n:n + 1])

        self.weights = numpy.multiply(self.weights, self.enabled_weights)

        error = distance_formula(target, self.node_values[:, self.middle_dem:])
        return error

    def save(self) -> str:
        weight_save = encode_list(self.weights, str, 0)
        enable_save = encode_list(self.enabled_weights, str, 0)
        save_string = "%d|%d|%d|%s|%s" % (self.in_dem - 1, self.out_dem, self.middle_dem, weight_save, enable_save)
        return save_string

    def load(self, save):
        save = save.split("|")
        in_dem = int(save[0])
        out_dem = int(save[1])
        middle_dem = int(save[2])

        weight_save = numpy.array(decode_list(save[3], float, 0))
        enable_save = decode_list(save[4], to_bool, 0)

        self.__init__(in_dem, out_dem, middle_dem, weights=weight_save, enabled_weights=enable_save)
