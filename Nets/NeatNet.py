import random
from typing import List, Tuple, Callable

import numpy

from Nets.LinearNet import LinearNet
from Nets.Net import Net
from SupportClasses.GeneticsPackage import GeneticsPackage
from formulas import sigmoid, sigmoid_der, color_formula, encode_gene, decode_gene

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
                 mutability_connections: float = 0.05,
                 mutability_nodes: float = 0.03,
                 mutability_reset: float = 0.1,
                 mutability_change_weight: float = 0.8,
                 mutability_toggle: float = 0.5,
                 excess_weight: float = 1.0,
                 disjoint_weight: float = 1.0,
                 weight_weight: float = 0.4,
                 weight_range: Tuple[float, float] = None,
                 enabled_weights: List[List[bool]] = None,
                 activation: Callable = sigmoid,
                 activation_der: Callable = sigmoid_der,
                 color_formula_param: Callable = color_formula,
                 weights: List[List[float]] = None
                 ):

        in_dem += 1
        self.genetics_package: GeneticsPackage = genetics_package
        self.connection_genes: List[Tuple[int, int, int, float, bool]] = connection_genes
        self.connection_genes.sort()

        self.mutability_weights = mutability_weights
        self.mutability_nodes = mutability_nodes
        self.mutability_connections = mutability_connections
        self.mutability_reset = mutability_reset
        self.mutability_change_weight = mutability_change_weight
        self.mutability_toggle = mutability_toggle
        self.excess_weight: float = excess_weight
        self.disjoint_weight: float = disjoint_weight
        self.weight_weight: float = weight_weight

        self.nodes = []
        middle_dem: int = 0
        if connection_genes is not None:
            nodes = set()
            middles = set()
            for connection_gene in self.connection_genes:
                nodes.add(connection_gene[1])
                nodes.add(connection_gene[2])

                if connection_gene[1] >= in_dem:
                    middles.add(connection_gene[1])

                if connection_gene[2] > 0:
                    middles.add(connection_gene[2])

            list(map(nodes.add, range(in_dem)))
            list(map(nodes.add, range(-out_dem, 0)))
            self.nodes = list(nodes)
            self.nodes.sort()
            # print(self.nodes)
            middle_dem = len(middles)
            enabled_weights = numpy.zeros((in_dem + middle_dem, middle_dem + out_dem), dtype=bool)
            weights = numpy.zeros((in_dem + middle_dem, middle_dem + out_dem))
            for innovation_number, start_node, end_node, weight, enabled in self.connection_genes:
                if enabled:
                    start = self.nodes.index(start_node) - out_dem
                    end = end_node if end_node < 0 else self.nodes.index(end_node) - (in_dem + out_dem)
                    enabled_weights[start][end] = True
                    weights[start][end] = weight

        elif weights:
            middle_dem = len(weights) - in_dem
        elif enabled_weights:
            middle_dem = len(enabled_weights) - in_dem
        LinearNet.__init__(
            self=self,
            in_dem=in_dem - 1,
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
        # if not (self.in_dem == net.in_dem and self.out_dem == net.out_dem and isinstance(net, NeatNet)):
        # print(self.in_dem == net.in_dem)
        # print(self.out_dem == net.out_dem)
        # print( isinstance(net, NeatNet))

        return self.in_dem == net.in_dem and self.out_dem == net.out_dem and isinstance(net, NeatNet)

    def breed(self, net: Net):
        assert self.compatible(net), isinstance(net, NeatNet)
        new_genes = self.cross_over(net)
        # print(new_genes)
        nodes = set()
        for n in range(self.in_dem):
            nodes.add(n)
        for n in range(1, self.out_dem + 1):
            nodes.add(-n)
        for connection_gene in self.connection_genes:
            if connection_gene[1] >= self.in_dem:
                nodes.add(connection_gene[1])

            if connection_gene[2] > 0:
                nodes.add(connection_gene[2])
        nodes = list(nodes)
        # print(nodes)
        nodes.sort()

        for index in range(len(new_genes)):
            if random.random() < self.mutability_change_weight:
                if random.random() < self.mutability_reset:
                    self.random_weight(index, new_genes)
                else:
                    self.shift_weight(index, new_genes)
            elif random.random() < self.mutability_toggle:
                self.toggle_connection(index, new_genes)
        # available = list(range(len(new_genes)))
        # while random.random() < self.mutability_nodes and available:
        #     gene = random.choice(available)
        #     available.remove(gene)
        #     self.add_node(gene, new_genes)

        available = random.shuffle(list(range(len(new_genes))))
        while random.random() < self.mutability_nodes and available:
            gene = available.pop(0)
            self.add_node(gene, new_genes)

        pairs = [(gene[START_INDEX], gene[END_INDEX]) for gene in self.connection_genes]
        # print(pairs)
        pair_choices = [(start, end) for end in nodes[:self.out_dem] + nodes[self.in_dem + self.out_dem:]
                        for start in nodes[self.out_dem:] if start != end]
        # print(pair_choices)
        list(map(pair_choices.remove, pairs))
        random.shuffle(pair_choices)
        # print(pair_choices)
        while random.random() < self.mutability_connections and pair_choices:
            # print("NEW!")
            pair = pair_choices.pop(0)
            self.add_connection_random(pair[0], pair[1], new_genes)

        new_net = NeatNet(
            self.in_dem - 1,
            self.out_dem,
            new_genes,
            self.genetics_package,
            self.mutability_weights,
            self.mutability_connections,
            self.mutability_nodes,
            self.mutability_reset,
            self.mutability_change_weight,
            self.mutability_toggle,
            self.excess_weight,
            self.disjoint_weight,
            self.weight_weight,
            activation=self.activation_function,
            activation_der=self.activation_derivative
        )
        # print("Done")
        return new_net

    def replicate(self):
        new_genes = self.connection_genes.copy()
        nodes = set()
        for n in range(self.in_dem):
            nodes.add(n)
        for n in range(1, self.out_dem + 1):
            nodes.add(-n)
        for connection_gene in self.connection_genes:
            # print(connection_gene)
            if connection_gene[1] >= self.in_dem:
                nodes.add(connection_gene[1])
                # print(connection_gene[1])

            if connection_gene[2] > 0:
                nodes.add(connection_gene[2])
                # print(connection_gene[2])
        nodes = list(nodes)
        nodes.sort()

        for index in range(len(new_genes)):
            if random.random() < self.mutability_change_weight:
                if random.random() < self.mutability_reset:
                    self.random_weight(index, new_genes)
                else:
                    self.shift_weight(index, new_genes)
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
        # print(nodes)
        pairs = [(gene[START_INDEX], gene[END_INDEX]) for gene in self.connection_genes]
        pair_choices = [(start, end) for end in nodes[:self.out_dem] + nodes[self.in_dem + self.out_dem:]
                        for start in nodes[self.out_dem:] if start != end]

        list(map(pair_choices.remove, pairs))
        pair_choices = random.shuffle(pair_choices)
        while random.random() < self.mutability_connections and pair_choices:
            pair = pair_choices.pop(0)
            self.add_connection_random(pair[0], pair[1], new_genes)

        new_net = NeatNet(
            self.in_dem - 1,
            self.out_dem,
            new_genes,
            self.genetics_package,
            self.mutability_weights,
            self.mutability_connections,
            self.mutability_nodes,
            self.mutability_reset,
            self.mutability_change_weight,
            self.mutability_toggle,
            self.excess_weight,
            self.disjoint_weight,
            self.weight_weight,
            activation=self.activation_function,
            activation_der=self.activation_derivative
        )
        return new_net

    def shift_weight(self, gene_index: int, genes: List[Tuple[int, int, int, float, bool]]):
        gene = genes[gene_index]
        gene = gene[0], gene[1], gene[2], gene[WEIGHT_INDEX] * self.mutability_weights * random.random(), gene[4],
        genes[gene_index] = gene
        assert not has_dup_gene(genes)

    def random_weight(self, gene_index: int, genes: List[Tuple[int, int, int, float, bool]]):
        gene = genes[gene_index]
        gene = gene[0], gene[1], gene[2], (2.0 * self.mutability_weights) * random.random() - self.mutability_weights, \
               gene[4]
        genes[gene_index] = gene
        assert not has_dup_gene(genes)

    def toggle_connection(self, gene_index: int, genes: List[Tuple[int, int, int, float, bool]]):
        gene = genes[gene_index]
        gene = gene[0], gene[1], gene[2], gene[3], not gene[4]
        genes[gene_index] = gene
        assert not has_dup_gene(genes)

    def disable_connection(self, gene_index: int, genes: List[Tuple[int, int, int, float, bool]]):
        gene = genes[gene_index]
        gene = gene[0], gene[1], gene[2], gene[3], False
        genes[gene_index] = gene
        assert not has_dup_gene(genes)

    def enable_connection(self, gene_index: int, genes: List[Tuple[int, int, int, float, bool]]):
        gene = genes[gene_index]
        gene = gene[0], gene[1], gene[2], gene[3], True
        genes[gene_index] = gene
        assert not has_dup_gene(genes)

    def add_node(self, gene_index: int, genes: List[Tuple[int, int, int, float, bool]]):
        old_gene = genes[gene_index]
        old_gene = old_gene[0], old_gene[1], old_gene[2], old_gene[3], False
        genes[gene_index] = old_gene
        node_innovation_number = self.genetics_package.add_node(old_gene[INNOVATION_INDEX])
        self.add_connection_weight(old_gene[START_INDEX], node_innovation_number, 1.0, genes)
        self.add_connection_weight(node_innovation_number, old_gene[END_INDEX], old_gene[WEIGHT_INDEX], genes)
        assert not has_dup_gene(genes)

    def add_connection_random(self, start_node, end_node, genes: List[Tuple[int, int, int, float, bool]]):
        innovation_number = self.genetics_package.add_connection(start_node, end_node)
        if not any([gene[INNOVATION_INDEX] == innovation_number for gene in genes]):
            genes.append((innovation_number, start_node, end_node, random.random() * 4.0 - 2.0, True))
        assert not has_dup_gene(genes)

    def add_connection_weight(self, start_node, end_node, weight, genes: List[Tuple[int, int, int, float, bool]]):
        innovation_number = self.genetics_package.add_connection(start_node, end_node)
        if not any([gene[INNOVATION_INDEX] == innovation_number for gene in genes]):
            genes.append((innovation_number, start_node, end_node, weight, True))
        assert not has_dup_gene(genes)

    def cross_over(self, net) -> List[Tuple[int, int, int, float, bool]]:
        new_genes = []
        index_self = 0
        index_net = 0
        while index_self < len(self.connection_genes) and index_net < len(net.connection_genes):
            if self.connection_genes[index_self][INNOVATION_INDEX] == net.connection_genes[index_net][INNOVATION_INDEX]:
                new_genes.append(random.choice((self.connection_genes[index_self], net.connection_genes[index_net])))
                index_net += 1
                index_self += 1
            elif self.connection_genes[index_self][INNOVATION_INDEX] < \
                    net.connection_genes[index_net][INNOVATION_INDEX]:
                new_genes.append(self.connection_genes[index_self])
                index_self += 1
            else:
                new_genes.append(net.connection_genes[index_net])
                index_net += 1
        new_genes.extend(self.connection_genes[index_self:])
        assert not has_dup_gene(new_genes)
        return new_genes

    def distance(self, net):
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
        geneome_size = max(len(self.connection_genes), len(net.connection_genes)) + 1
        # print("(%f, %f ,%f)" % (weight_diff, disjoint_diff, excess_diff))
        # print("(%f, %f ,%f)" % (self.weight_weight * weight_diff / geneome_size,
        #                         self.disjoint_weight * disjoint_diff / geneome_size,
        #                         self.excess_weight * excess_diff / geneome_size))
        return self.weight_weight * weight_diff / geneome_size + \
               self.disjoint_weight * disjoint_diff / geneome_size + \
               self.excess_weight * excess_diff / geneome_size

    def save(self) -> str:
        genes_string = ",".join(map(encode_gene, self.connection_genes))
        save_string = "%d|%d|%f|%f|%f|%f|%f|%f|%s" % (self.in_dem - 1,
                                                      self.out_dem,
                                                      self.mutability_weights,
                                                      self.mutability_connections,
                                                      self.mutability_nodes,
                                                      self.mutability_reset,
                                                      self.mutability_change_weight,
                                                      self.mutability_toggle,
                                                      genes_string)
        return save_string

    def load(self, save):
        save = save.split("|")
        in_dem = int(save[0])
        out_dem = int(save[1])
        mutability_weights: float = float(save[2])
        mutability_connections: float = float(save[3])
        mutability_nodes: float = float(save[4])
        mutability_reset: float = float(save[5])
        mutability_change_weight: float = float(save[6])
        mutability_toggle: float = float(save[7])
        gene_string = save[8]
        genes = list(map(decode_gene, gene_string.split(",")))
        self.__init__(in_dem, out_dem, genes, self.genetics_package,
                      mutability_connections=mutability_connections,
                      mutability_nodes=mutability_nodes,
                      mutability_toggle=mutability_toggle,
                      mutability_change_weight=mutability_change_weight,
                      mutability_reset=mutability_reset,
                      mutability_weights=mutability_weights)

    def __str__(self):
        return "(<%d, %d> [%s])" % (self.in_dem, self.out_dem, ",".join(map(str, self.connection_genes)))


def has_dup_gene(genes):
    # print(",".join(map(str, genes)))
    return any([i != j and genes[i][INNOVATION_INDEX] == genes[j][INNOVATION_INDEX] for i in range(len(genes))
                for j in range(len(genes))])
