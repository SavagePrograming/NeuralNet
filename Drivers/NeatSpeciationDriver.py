import random
from typing import ClassVar, List

import pygame
from math import ceil
from numpy import mean, median, math

from Drivers.Driver import Driver
from Drivers.EvolutionSpeciationDriver import EvolutionSpeciationDriver
from Nets.EvolvingNet import EvolvingNet
from Nets.NeatNet import NeatNet
from Simulations.EvolutionSimulation import EvolutionSimulation
from SupportClasses.GeneticsPackage import GeneticsPackage


def clean(a):
    i = 0
    while i < len(a):
        if not a[i]:
            del a[i]
        else:
            i += 1


class NeatSpeciationDriver(EvolutionSpeciationDriver):
    def __init__(self,
                 population_size: int,
                 survivor_ratio: float,
                 simulation: EvolutionSimulation,
                 generation_size: int,
                 species_threshold: float,
                 species_independent_survivor_ratio: float,
                 balancing_focus: float,
                 mutability_weights=2.0,
                 mutability_connections=0.05,
                 mutability_nodes=0.03,
                 mutability_reset=0.1,
                 mutability_change_weight=0.8,
                 mutability_toggle=0.1,
                 excess_weight: float = 1.0,
                 disjoint_weight: float = 1.0,
                 weight_weight: float = 0.4,
                 inter_species_breeding_rate=0.001,
                 asexual_breading_rate=.25,
                 draw_count = 3):

        self.in_dem: int = simulation.in_dem
        self.out_dem: int = simulation.out_dem
        self.row_size: int = 0
        self.row_count: int = 0
        self.population_size: int = population_size
        self.simulation = simulation
        self.evolving_class = NeatNet
        self.generation_size: int = generation_size
        self.median: float = 0.0
        self.average: float = 0.0
        self.maximum: float = 0.0
        self.minimum: float = 0.0

        self.balance_top: float = .5
        self.balance_bottom: float = .25
        self.species_threshold: float = species_threshold
        self.balance_focus: float = balancing_focus
        self.SISR: float = species_independent_survivor_ratio
        self.survivor_ratio: float = survivor_ratio
        self.inter_species_breeding_rate = inter_species_breeding_rate
        self.asexual_breading_rate = asexual_breading_rate
        self.species: List[List[NeatNet]] = []
        self.draw_count = draw_count

        self.gene_pool = GeneticsPackage(self.in_dem, self.out_dem)

        self.population: List[NeatNet] = []
        for i in range(self.population_size):
            self.population.append(self.evolving_class(self.in_dem, self.out_dem, [], self.gene_pool,
                                                       mutability_weights=mutability_weights,
                                                       mutability_connections=mutability_connections,
                                                       mutability_nodes=mutability_nodes,
                                                       mutability_reset=mutability_reset,
                                                       mutability_change_weight=mutability_change_weight,
                                                       mutability_toggle=mutability_toggle,
                                                       weight_weight=weight_weight,
                                                       disjoint_weight=disjoint_weight,
                                                       excess_weight=excess_weight))

        for child in self.population:
            self.add_to_specie(child)
        # print(len(self.species))

    # def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, dot_size: int = 10):
    #
    #     self.row_count = int(math.sqrt(self.draw_count))
    #     self.row_size = math.ceil(self.draw_count/ self.row_count)
    #     # print("-------------")
    #     # print("POP " + ",".join(map(str, self.population)))
    #     diff = math.ceil(self.population_size / self.draw_count + 1)
    #     for i in range(self.row_count * self.row_size):
    #         if i * diff < self.population_size:
    #             self.population[i * diff].update(screen,
    #                                       x + (i % self.row_size) * (width // self.row_size),
    #                                       y + (i // self.row_size) * (height // self.row_count),
    #                                       width // self.row_size,
    #                                       height // self.row_count,
    #                                       dot_size)
    #             self.population[i * diff].draw()
    #             # print(self.species[i][0])
    #             # print(self.species[i][0].input_nodes)
    #             # print(self.species[i][0].in_color_range)