from functools import total_ordering

import numpy, random, math, pygame

from formulas import distance_formula, sigmoid, sigmoid_der, randomize


def color_formula(x):
    return [0, int(x * 255.), 0]


class LinearNet:
    def __init__(self, in_dem, out_dem, middle_dem, weight_range=[2.0, -2.0],
                 enabled_weights=None,
                 activation=sigmoid,
                 activation_der=sigmoid_der,
                 color_formula_param=color_formula, weights=None):

        self.in_dem = in_dem + 1
        self.out_dem = out_dem
        self.middle_dem = middle_dem

        self.Score = 0
        self.input_nodes = numpy.zeros((1, self.in_dem))

        self.node_values = numpy.zeros((1, self.middle_dem + self.out_dem))
        self.node_sum = numpy.zeros((1, self.middle_dem + self.out_dem))
        self.node_back = numpy.zeros((1, self.in_dem + self.middle_dem))
        if weights:
            self.weights = weights
        else:
            dif = abs(weight_range[0] - weight_range[1])
            self.weights = randomize(numpy.zeros((self.in_dem + self.middle_dem, self.middle_dem + self.out_dem)))
            self.weights = numpy.add(weight_range[1], numpy.multiply(dif, self.weights))

        if enabled_weights:
            self.enabled_weights = enabled_weights
        else:
            self.enabled_weights = numpy.array([
                [in_node < out_node
                 for out_node in range(self.in_dem, self.in_dem + self.middle_dem + self.out_dem)]
                for in_node in range(self.in_dem + self.middle_dem)])

        self.activation_function = activation
        self.activation_derivative = activation_der

        self.color_formula = color_formula_param

    def set_in(self, array):
        array = array + [1.0]
        assert len(array) == self.in_dem
        self.input_nodes = numpy.array(array, ndmin=2)

    def get_out(self):
        # print("LINEAR INPUT:" + str(self.input_nodes))
        self.weights = numpy.multiply(self.weights, self.enabled_weights)
        numpy.zeros((1, self.middle_dem + self.out_dem))
        self.node_sum = numpy.dot(self.input_nodes, self.weights[:self.in_dem])
        # print("LINEAR SUM 1:" + str(self.node_sum))
        for i in range(self.middle_dem):
            self.node_values[0][i] = self.activation_function(self.node_sum[0][i])
            # print("LINEAR IN:" + str(self.node_values[0][i]))
            self.node_sum = numpy.add(self.node_sum, numpy.multiply(self.node_values[0][i],
                                                                    self.weights[self.in_dem + i:self.in_dem + i + 1]))
            # print("LINEAR SUM "+str(i+2)+":" + str(self.node_sum))
        self.node_values = self.activation_function(self.node_sum)

        # print("LINEAR OUT:" + str(self.node_values[0][self.middle_dem:]))
        return self.node_values[0][self.middle_dem:]

    def learn(self, ratio, target):
        self.weights = numpy.multiply(self.weights, self.enabled_weights)
        target = numpy.reshape(target, (1, len(target)))
        # print("LINEAR TARGET" + str(target))
        # print("LINEAR ACT:" + str(self.node_values[:, self.middle_dem:]))

        difference = numpy.multiply(2.0, numpy.subtract(target, self.node_values[:, self.middle_dem:]))
        # print("LINEAR DIFF:" + str(difference))
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
            # print("+++++")
            # print(weight_shift)

            self.weights[:, n:n + 1] = numpy.add(weight_shift, self.weights[:, n:n + 1])

            # print("+++++")
        # print("Past:" + str(self.node_back))
        self.weights = numpy.multiply(self.weights, self.enabled_weights)

        error = distance_formula(target, self.node_values[:, self.middle_dem:])
        return error

    def draw(self, screen, x, y, width, height, scale_dot=5):

        in_spacing = (height - scale_dot * 2) / (self.in_dem + 1)

        center_x = x + width / 2  # x + width / 4
        center_y = y + height - scale_dot
        radius_y = height - scale_dot * 2
        radius_x = width / 4  # width / 2
        angle = 0
        if self.middle_dem > 1:
            angle = (math.pi) / (self.middle_dem - 1)

        color_range = [self.color_formula(in_node) for in_node in self.input_nodes[0]]

        in_range_loc = numpy.zeros((self.in_dem, 2)).astype(int)
        in_range_loc[:, 0:1] = (numpy.add(x + scale_dot, in_range_loc[:, 0:1]))
        in_range_loc[:, 1:2] = (numpy.add(y + scale_dot, numpy.multiply(
            numpy.add(numpy.reshape(range(self.in_dem), (self.in_dem, 1)), 1), in_spacing)))

        [pygame.draw.circle(screen, color, pos, scale_dot) for color, pos in zip(color_range, in_range_loc)]
        # (math.pi / 2) +

        # for i in range(self.middle_dem):
        #     helper3(i)
        def helper_draw_2(i, j):
            if self.enabled_weights[j][i]:
                pygame.draw.line(screen,
                                 [255. - 255. * self.activation_function(self.weights[j][i]), 125
                                     , 255. * self.activation_function(self.weights[j][i])],
                                 [int(x + scale_dot), int(y + scale_dot + (j + 1) * in_spacing)],
                                 [int(center_x - math.cos(angle * i) * radius_x - scale_dot / 2),
                                  int(center_y - math.sin(angle * i) * radius_y - scale_dot / 2)])
        def helper_draw_1(i):
            list(map(helper_draw_2, [i] * self.in_dem, range(self.in_dem)))

        def helper_draw_4(i,j):
            if self.enabled_weights[j + self.in_dem][i]:
                pygame.draw.line(screen,
                                 [255. - 255. * self.activation_function(self.weights[j + self.in_dem][i]), 125
                                     , 255. * self.activation_function(self.weights[j + self.in_dem][i])],
                                 [int(center_x - math.cos(angle * j) * radius_x + scale_dot / 2),
                                  int(center_y - math.sin(angle * j) * radius_y + scale_dot / 2)],
                                 [int(center_x - math.cos(angle * i) * radius_x - scale_dot / 2),
                                  int(center_y - math.sin(angle * i) * radius_y - scale_dot / 2)])
        def helper_draw_3(i):
            for j in range(self.middle_dem):
                helper_draw_4(i, j)
            pygame.draw.circle(screen,
                               self.color_formula(self.node_values[0][i]),
                               [int(center_x - math.cos(angle * i) * radius_x),
                                int(center_y - math.sin(angle * i) * radius_y)],
                               int(scale_dot))
        list(map(helper_draw_1, range(self.middle_dem)))
        list(map(helper_draw_3, range(self.middle_dem)))
        out_spacing = (height - scale_dot * 2) / (self.out_dem + 1)
        for i in range(self.out_dem):
            for j in range(self.in_dem):
                if self.enabled_weights[j][self.middle_dem + i]:
                    pygame.draw.line(screen,
                                     [255. - 255. * self.activation_function(self.weights[j][self.middle_dem + i]), 125
                                         , 255. * self.activation_function(self.weights[j][self.middle_dem + i])],
                                     [int(x + scale_dot), int(y + scale_dot + (j + 1) * in_spacing)],
                                     [int(x + width - scale_dot),
                                      int(y + scale_dot + (i + 1) * out_spacing)])
            for j in range(self.middle_dem):
                if self.enabled_weights[j + self.in_dem][self.middle_dem + i]:
                    pygame.draw.line(screen,
                                     [255. - 255. * self.activation_function(
                                         self.weights[j + self.in_dem][self.middle_dem + i]), 125
                                         , 255. * self.activation_function(
                                         self.weights[j + self.in_dem][self.middle_dem + i])],
                                     [int(center_x - math.cos(angle * j) * radius_x + scale_dot / 2),
                                      int(center_y - math.sin(angle * j) * radius_y + scale_dot / 2)],
                                     [int(x + width - scale_dot),
                                      int(y + scale_dot + (i + 1) * out_spacing)])
            pygame.draw.circle(screen,
                               self.color_formula(self.node_values[0][self.middle_dem + i]),
                               [int(x + width - scale_dot),
                                int(y + scale_dot + (i + 1) * out_spacing)],
                               int(scale_dot))

    def __eq__(self, other):
        return self.Score == other.Score

    def __lt__(self, other):
        return self.Score < other.Score

    def __gt__(self, other):
        return self.Score > other.Score

    def __ge__(self, other):
        return self.Score >= other.Score

    def __le__(self, other):
        return self.Score <= other.Score

    def __add__(self, other):
        if isinstance(other, LinearNet):
            return self.Score + other.Score
        else:
            return self.Score + other

    def __radd__(self, other):
        if isinstance(other, LinearNet):
            return self.Score + other.Score
        else:
            return self.Score + other
