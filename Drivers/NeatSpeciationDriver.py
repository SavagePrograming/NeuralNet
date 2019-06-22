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
                 mutability_weights=0.5,
                 mutability_connections=0.5,
                 mutability_nodes=0.5,
                 mutability_reset=0.5,
                 mutability_shift=0.5,
                 mutability_toggle=0.5):

        self.in_dem: int = simulation.in_dem
        self.out_dem: int = simulation.in_dem
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
        self.species: List[List[NeatNet]] = []

        self.gene_pool = GeneticsPackage(self.in_dem, self.out_dem)

        self.population: List[NeatNet] = []
        for i in range(self.population_size):
            self.population.append(self.evolving_class(self.in_dem, self.out_dem, [], self.gene_pool,
                                                       mutability_weights=mutability_weights,
                                                       mutability_connections=mutability_connections,
                                                       mutability_nodes=mutability_nodes,
                                                       mutability_reset=mutability_reset,
                                                       mutability_shift=mutability_shift,
                                                       mutability_toggle=mutability_toggle))

        for child in self.population:
            self.add_to_specie(child)
