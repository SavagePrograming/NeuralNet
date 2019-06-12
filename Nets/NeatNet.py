from typing import List, Tuple, Callable

import numpy

from Nets.LinearNet import LinearNet
from formulas import sigmoid, sigmoid_der, color_formula


class InvalidInnovationNumber(LookupError):
    def __init__(self, innovation_number, is_node):
        super(InvalidInnovationNumber, self).__init__(
            "Inovation number '%d' which is a %s" % (innovation_number, "node" if is_node else "connection"))


class GeneticsPackage():
    def __init__(self, in_dem: int, out_dem: int):
        in_dem += 1
        self.connection_genes = [[0] * (out_dem) for y in range(in_dem)]
        self.node_genes = {}
        self.connection_innovation_number = 1
        self.node_innovation_number = in_dem
        self.in_dem = in_dem
        self.out_dem = out_dem

    def add_connection(self, start_node: int, end_node: int) -> int:
        if 0 < start_node < self.node_innovation_number:
            if end_node < self.node_innovation_number and (max(start_node, self.in_dem - 1) < end_node or end_node < 0):
                end_node = end_node if end_node < 0 else end_node - self.in_dem
                if self.connection_genes[start_node][end_node] == 0:
                    self.connection_genes[start_node][end_node] = self.connection_innovation_number
                    self.connection_innovation_number += 1
                    return self.connection_innovation_number - 1
                else:
                    return self.connection_genes[start_node][end_node]
            else:
                raise InvalidInnovationNumber(end_node, True)
        else:
            raise InvalidInnovationNumber(start_node, True)

    def add_node(self, connection_inovation_number: int) -> int:
        if 0 < connection_inovation_number < self.connection_innovation_number:
            if connection_inovation_number in self.node_genes:
                return self.node_genes[connection_inovation_number]
            else:
                self.node_genes[connection_inovation_number] = self.node_innovation_number
                for x in range(len(self.connection_genes)):
                    self.connection_genes[x].insert(self.node_innovation_number, 0)
                self.connection_genes.insert(self.node_innovation_number,
                                             [0] * (self.out_dem + self.node_innovation_number + 1))
                self.node_innovation_number += 1
                return self.node_innovation_number - 1
        raise InvalidInnovationNumber(connection_inovation_number, True)

    def get_connection_innovation(self, start_node: int, end_node: int) -> int:
        if 0 < start_node < self.node_innovation_number:
            if end_node < self.node_innovation_number and \
                    (max(start_node, self.in_dem - 1) < end_node or end_node < 0):
                end_node = end_node if end_node < 0 else end_node - self.in_dem
                if self.connection_genes[start_node][end_node] == 0:
                    raise InvalidInnovationNumber(0, False)
                else:
                    return self.connection_genes[start_node][end_node]
            else:
                raise InvalidInnovationNumber(end_node, True)
        else:
            raise InvalidInnovationNumber(start_node, True)

    def get_node_innovation(self, connection_inovation_number: int) -> int:
        if connection_inovation_number in self.node_genes:
            return self.node_genes[connection_inovation_number]
        raise InvalidInnovationNumber(connection_inovation_number, False)


class NeatNet(LinearNet):
    def __init__(self,
                 in_dem: int,
                 out_dem: int,
                 connection_genes: List[Tuple[int, int, int, float, bool]],
                 # (innovation_number, start_node, end_node, weight, enabled)
                 genetics_package: GeneticsPackage,
                 weight_range: Tuple[float, float] = None,
                 enabled_weights: List[List[bool]] = None,
                 activation: Callable = sigmoid,
                 activation_der: Callable = sigmoid_der,
                 color_formula_param: Callable = color_formula,
                 weights: List[List[float]] = None
                 ):
        self.genetics_package = genetics_package
        self.connection_genes = connection_genes
        middle_dem = 0
        if connection_genes:
            nodes = set()
            for connection_gene in self.connection_genes:
                if connection_gene[1] >= self.in_dem:
                    nodes.add(connection_genes[1])

                if connection_gene[1] > 0:
                    nodes.add(connection_genes[2])
            nodes = list(nodes)
            middle_dem = len(nodes)
            enabled_weights = numpy.zeros((in_dem + middle_dem, middle_dem + out_dem), dtype=bool)
            weights = numpy.zeros((in_dem + middle_dem, middle_dem + out_dem))
            for innovation_number, start_node, end_node, weight, enabled in self.connection_genes:
                if enabled:
                    start = start_node if start_node < self.in_dem else nodes.index(start_node)
                    end = end_node if end_node < 0 else nodes.index(start_node)
                    enabled_weights[start][end] = True
                    weights[start][end] = weight

        elif weights:
            middle_dem = len(weights) - in_dem
        elif enabled_weights:
            middle_dem = len(enabled_weights) - in_dem

        LinearNet.__init__(
            self=self,
            in_dem=in_dem,
            out_dem=out_dem,
            middle_dem=middle_dem,
            weight_range=weight_range,
            enabled_weights=enabled_weights,
            activation=activation,
            activation_der=activation_der,
            color_formula_param=color_formula_param,
            weights=weights,
        )

    def compatable(self, net):
        pass

    def breed(self, net):
        pass

    def replicate(self):
        pass

    def mutate(self, number):
        pass

    def distance(self, net):
        pass
