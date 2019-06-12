import numpy, math, pygame

from formulas import distance_formula, sigmoid, sigmoid_der, randomize


def color_formula(x):
    return [0, int(x * 255.), 0]


def color_formula_line(x):
    return [255. - 255. * sigmoid(x), 125, 255. * sigmoid(x)]


def color_formula_line_helper(x):
    return list(map(color_formula_line, x))


def draw_circle(screen_range, color_range, in_range_loc, radius_range):
    pygame.draw.circle(screen_range, color_range, in_range_loc, radius_range)


def draw_line_enable(screen, color, start_pos, end_pos, width, enable):
    if enable:
        pygame.draw.line(screen, color, start_pos, end_pos, width)


def draw_line_helper(screen, color, start_pos, end_pos, width, enable):
    list(map(draw_line_enable, screen, color, start_pos, end_pos, width, enable))


class LinearNet:
    def __init__(self,
                 in_dem,
                 out_dem,
                 middle_dem,
                 weight_range=[2.0, -2.0],
                 enabled_weights=None,
                 activation=sigmoid,
                 activation_der=sigmoid_der,
                 color_formula_param=color_formula,
                 weights=None):

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
    def update(self, screen, x, y, width, height, scale_dot=5):

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


    def set_in(self, array):
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

    def learn(self, ratio, target):
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
