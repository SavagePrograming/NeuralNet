from typing import List, Dict

from SupportClasses.InvalidInnovationNumber import InvalidInnovationNumber


class GeneticsPackage:
    def __init__(self,
                 in_dem: int,
                 out_dem: int):
        in_dem += 1
        self.connection_genes: List[List[int]] = [[0] * (out_dem) for y in range(in_dem)]
        self.node_genes: Dict[int, int] = {}
        self.connection_innovation_number: int = 1
        self.node_innovation_number: int = in_dem
        self.in_dem: int = in_dem
        self.out_dem: int = out_dem

    def add_connection(self, start_node: int, end_node: int) -> int:
        if 0 <= start_node < self.node_innovation_number:
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

    def add_node(self, connection_innovation_number: int) -> int:
        if 0 < connection_innovation_number < self.connection_innovation_number:
            if connection_innovation_number in self.node_genes:
                return self.node_genes[connection_innovation_number]
            else:
                self.node_genes[connection_innovation_number] = self.node_innovation_number
                for x in range(len(self.connection_genes)):
                    self.connection_genes[x].insert(self.node_innovation_number, 0)
                self.connection_genes.insert(self.node_innovation_number,
                                             [0] * (self.out_dem + self.node_innovation_number + 1))
                self.node_innovation_number += 1
                return self.node_innovation_number - 1
        raise InvalidInnovationNumber(connection_innovation_number, True)

    def get_connection_innovation(self, start_node: int, end_node: int) -> int:
        if 0 <= start_node < self.node_innovation_number:
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

    def get_node_innovation(self, connection_innovation_number: int) -> int:
        if connection_innovation_number in self.node_genes:
            return self.node_genes[connection_innovation_number]
        raise InvalidInnovationNumber(connection_innovation_number, False)