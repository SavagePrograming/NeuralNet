import random
from typing import List, Tuple, Callable, Dict

import numpy

from Nets.LinearNet import LinearNet
from Nets.Net import Net
from SupportClasses.GeneticsPackage import GeneticsPackage
from formulas import sigmoid, sigmoid_der, color_formula, discrete_tests

# (innovation_number, start_node, end_node, weight, enabled)
INNOVATION_INDEX = 0
START_INDEX = 1
END_INDEX = 2
WEIGHT_INDEX = 3
ENABLED_INDEX = 4


class NeatNet(LinearNet):
    def __init__(self,
                 in_dem: int,
                 out_dem: int,
                 connection_genes: List[Tuple[int, int, int, float, bool]],
                 # (innovation_number, start_node, end_node, weight, enabled)
                 genetics_package: GeneticsPackage,
                 mutability_weights: float = 2.0,
                 mutability_connections: float = 0.5,
                 mutability_nodes: float = 0.5,
                 mutability_reset: float = 0.5,
                 mutability_shift: float = 0.5,
                 mutability_toggle: float = 0.5,
                 weight_range: Tuple[float, float] = None,
                 enabled_weights: List[List[bool]] = None,
                 activation: Callable = sigmoid,
                 activation_der: Callable = sigmoid_der,
                 color_formula_param: Callable = color_formula,
                 weights: List[List[float]] = None
                 ):

        self.genetics_package: GeneticsPackage = genetics_package
        self.connection_genes: List[Tuple[int, int, int, float, bool]] = connection_genes
        self.connection_genes.sort()

        self.mutability_weights = mutability_weights
        self.mutability_nodes = mutability_nodes
        self.mutability_connections = mutability_connections
        self.mutability_reset = mutability_reset
        self.mutability_shift = mutability_shift
        self.mutability_toggle = mutability_toggle

        self.nodes = []
        middle_dem: int = 0
        if connection_genes:
            nodes = set()
            for connection_gene in self.connection_genes:
                if connection_gene[1] >= self.in_dem:
                    nodes.add(connection_genes[1])

                if connection_gene[1] > 0:
                    nodes.add(connection_genes[2])
            self.nodes = list(nodes)
            middle_dem = len(self.nodes)
            enabled_weights = numpy.zeros((in_dem + middle_dem, middle_dem + out_dem), dtype=bool)
            weights = numpy.zeros((in_dem + middle_dem, middle_dem + out_dem))
            for innovation_number, start_node, end_node, weight, enabled in self.connection_genes:
                if enabled:
                    start = start_node if start_node < self.in_dem else self.nodes.index(start_node)
                    end = end_node if end_node < 0 else self.nodes.index(start_node)
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

    def compatible(self, net: Net):
        return self.in_dem == net.in_dem and self.out_dem == net.out_dem and isinstance(net, NeatNet)

    def breed(self, net: NeatNet):
        assert self.compatible(net)
        new_genes = self.cross_over(net)
        nodes = set()
        for connection_gene in self.connection_genes:
            if connection_gene[1] >= self.in_dem:
                nodes.add(new_genes[1])

            if connection_gene[1] > 0:
                nodes.add(new_genes[2])
        nodes = list(nodes)
        nodes.sort()

        for index in range(len(new_genes)):
            if random.random() < self.mutability_shift:
                self.shift_weight(index, new_genes)
            elif random.random() < self.mutability_reset:
                self.random_weight(index, new_genes)
            elif random.random() < self.mutability_toggle:
                self.toggle_connection(index, new_genes)
        available = list(range(len(new_genes)))
        while random.random() < self.mutability_nodes and available:
            gene = random.choice(available)
            available.remove(gene)
            self.add_node(gene, new_genes)

        available = random.shuffle(list(range(len(new_genes))))
        while random.random() < self.mutability_nodes and available:
            gene = available.pop(0)
            self.add_node(gene, new_genes)

        pairs = [(gene[START_INDEX], gene[END_INDEX]) for gene in new_genes]
        pair_choices = [[(start, end) for end in nodes[:self.out_dem] + nodes[self.in_dem + self.out_dem:]]
                        for start in nodes[self.out_dem:]]
        list(map(pair_choices.remove, pairs))
        pair_choices = random.shuffle(pair_choices)
        while random.random() < self.mutability_connections and pair_choices:
            pair = pair_choices.pop(0)
            self.add_connection_random(pair[0], pair[1], new_genes)

        new_net = NeatNet(
            self.in_dem,
            self.out_dem,
            new_genes,
            self.genetics_package,
            self.mutability_weights,
            self.mutability_connections,
            self.mutability_nodes,
            self.mutability_reset,
            self.mutability_shift,
            self.mutability_toggle,
            activation=self.activation_function,
            activation_der=self.activation_derivative
        )
        return new_net

    def replicate(self):
        new_genes = self.connection_genes
        nodes = set()
        for connection_gene in self.connection_genes:
            if connection_gene[1] >= self.in_dem:
                nodes.add(new_genes[1])

            if connection_gene[1] > 0:
                nodes.add(new_genes[2])
        nodes = list(nodes)
        nodes.sort()

        for index in range(len(new_genes)):
            if random.random() < self.mutability_shift:
                self.shift_weight(index, new_genes)
            elif random.random() < self.mutability_reset:
                self.random_weight(index, new_genes)
            elif random.random() < self.mutability_toggle:
                self.toggle_connection(index, new_genes)
        available = list(range(len(new_genes)))
        while random.random() < self.mutability_nodes and available:
            gene = random.choice(available)
            available.remove(gene)
            self.add_node(gene, new_genes)

        available = random.shuffle(list(range(len(new_genes))))
        while random.random() < self.mutability_nodes and available:
            gene = available.pop(0)
            self.add_node(gene, new_genes)

        pairs = [(gene[START_INDEX], gene[END_INDEX]) for gene in new_genes]
        pair_choices = [[(start, end) for end in nodes[:self.out_dem] + nodes[self.in_dem + self.out_dem:]]
                        for start in nodes[self.out_dem:]]
        list(map(pair_choices.remove, pairs))
        pair_choices = random.shuffle(pair_choices)
        while random.random() < self.mutability_connections and pair_choices:
            pair = pair_choices.pop(0)
            self.add_connection_random(pair[0], pair[1], new_genes)

        new_net = NeatNet(
            self.in_dem,
            self.out_dem,
            new_genes,
            self.genetics_package,
            self.mutability_weights,
            self.mutability_connections,
            self.mutability_nodes,
            self.mutability_reset,
            self.mutability_shift,
            self.mutability_toggle,
            activation=self.activation_function,
            activation_der=self.activation_derivative
        )
        return new_net

    def shift_weight(self, gene_index: int, genes: List[Tuple[int, int, int, float, bool]]):
        gene = genes[gene_index]
        gene = gene[0], gene[1], gene[2], gene[WEIGHT_INDEX] * self.mutability_weights * random.random(), gene[4],
        genes[gene_index] = gene

    def random_weight(self, gene_index: int, genes: List[Tuple[int, int, int, float, bool]]):
        gene = genes[gene_index]
        gene = gene[0], gene[1], gene[2], (2.0 * self.mutability_weights) * random.random() - self.mutability_weights, \
               gene[4]
        genes[gene_index] = gene

    def toggle_connection(self, gene_index: int, genes: List[Tuple[int, int, int, float, bool]]):
        gene = genes[gene_index]
        gene = gene[0], gene[1], gene[2], gene[3], not gene[4]
        genes[gene_index] = gene

    def disable_connection(self, gene_index: int, genes: List[Tuple[int, int, int, float, bool]]):
        gene = genes[gene_index]
        gene = gene[0], gene[1], gene[2], gene[3], False
        genes[gene_index] = gene

    def enable_connection(self, gene_index: int, genes: List[Tuple[int, int, int, float, bool]]):
        gene = genes[gene_index]
        gene = gene[0], gene[1], gene[2], gene[3], True
        genes[gene_index] = gene

    def add_node(self, gene_index: int, genes: List[Tuple[int, int, int, float, bool]]):
        old_gene = genes[gene_index]
        old_gene = old_gene[0], old_gene[1], old_gene[2], old_gene[3], False
        genes[gene_index] = old_gene
        node_innovation_number = self.genetics_package.get_node_innovation(old_gene[INNOVATION_INDEX])
        self.add_connection_weight(old_gene[START_INDEX], node_innovation_number, 1.0)
        self.add_connection_weight(node_innovation_number, old_gene[END_INDEX], old_gene[WEIGHT_INDEX])

    def add_connection_random(self, start_node, end_node, genes: List[Tuple[int, int, int, float, bool]]):
        innovation_number = self.genetics_package.get_connection_innovation(start_node, end_node)
        if not any([gene[INNOVATION_INDEX] == innovation_number for gene in genes]):
            genes.append((innovation_number, start_node, end_node, random.random() * 4.0 - 2.0, True))

    def add_connection_weight(self, start_node, end_node, weight, genes: List[Tuple[int, int, int, float, bool]]):
        innovation_number = self.genetics_package.get_connection_innovation(start_node, end_node)
        if not any([gene[INNOVATION_INDEX] == innovation_number for gene in genes]):
            genes.append((innovation_number, start_node, end_node, weight, True))

    def cross_over(self, net: NeatNet) -> List[Tuple[int, int, int, float, bool]]:
        new_genes = []
        index_self = 0
        index_net = 0
        while index_self < len(self.connection_genes) and index_net < len(net.connection_genes):
            if self.connection_genes[index_self][INNOVATION_INDEX] == net.connection_genes[index_net][INNOVATION_INDEX]:
                new_genes.append(random.choice((self.connection_genes[index_self], net.connection_genes[index_net])))
            elif self.connection_genes[index_self][INNOVATION_INDEX] < \
                    net.connection_genes[index_net][INNOVATION_INDEX]:
                new_genes.append(self.connection_genes[index_self])
                index_self += 1
            else:
                new_genes.append(net.connection_genes[index_net])
                index_net += 1
        new_genes.extend(self.connection_genes[index_self:])
        return new_genes

    def distance(self, net: NeatNet):
        assert self.compatible(net)
        weight_diff = 0.0
        disjoint_diff = 0
        excess_diff = 0
        index_self = 0
        index_net = 0
        while index_self < len(self.connection_genes) and index_net < len(net.connection_genes):
            if self.connection_genes[index_self][INNOVATION_INDEX] == net.connection_genes[index_net][INNOVATION_INDEX]:
                weight_diff += abs(
                    self.connection_genes[index_self][WEIGHT_INDEX] - net.connection_genes[index_net][WEIGHT_INDEX])
                index_self += 1
                index_net += 1
            elif self.connection_genes[index_self][INNOVATION_INDEX] < \
                    net.connection_genes[index_net][INNOVATION_INDEX]:
                disjoint_diff += 1
                index_self += 1
            else:
                disjoint_diff += 1
                index_net += 1
        excess_diff += len(self.connection_genes) - index_self + len(net.connection_genes) - index_net
    def save(self) -> str:
        weight_save = encode_list(self.weights, str, 0)
        enable_save = encode_list(self.enabled_weights, str, 0)
        save_string = "%d|%d|%d|%s|%s" % (self.in_dem - 1, self.out_dem, self.middle_dem, weight_save, enable_save)
        return save_string

    def load(self, save):
        save = save.split("|")
        in_dem = int(save[0])
        out_dem = int(save[1])
         mutability_weights: float = float(save[2])
         mutability_connections: float = float(save[3])
         mutability_nodes: float =float(save[4])
         mutability_reset: float = float(save[5])
         mutability_shift: float = float(save[6])
         mutability_toggle: float = float(save[7])


        self.__init__(in_dem, out_dem, weights=weight_save, enabled_weights=enable_save)
