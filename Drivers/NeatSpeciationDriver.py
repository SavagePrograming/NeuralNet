import random
from typing import ClassVar, List, Tuple

import pygame
from math import ceil
from numpy import mean, median, math

from Drivers.Driver import Driver
from Drivers.EvolutionSpeciationDriver import EvolutionSpeciationDriver
from Nets.EvolvingNet import EvolvingNet
from Nets.NeatNet import NeatNet
from Simulations.EvolutionSimulation import EvolutionSimulation
from SupportClasses import Specieator
from SupportClasses.GeneticsPackage import GeneticsPackage
from formulas import get_species


def clean(a, b):
    i = 0
    while i < len(a):
        if not a[i]:
            del a[i]
            del b[i]
        else:
            i += 1


class NeatSpeciationDriver(EvolutionSpeciationDriver):
    def __init__(self,
                 population_size: int,
                 reproducers_ratio: float,
                 simulation: EvolutionSimulation,
                 generation_size: int,
                 species_threshold: float,
                 balance_focus: float,
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
                 draw_count=3,
                 stagnant_limit=5):

        self.reprocucers = []
        self.stagnant_limit = stagnant_limit
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

        self.reproducer_ratio: float = reproducers_ratio
        self.inter_species_breeding_rate = inter_species_breeding_rate
        self.asexual_breading_rate = asexual_breading_rate
        self.draw_count = draw_count
        self.generation_count: int = 0
        self.specieator = Specieator.Specieator(species_threshold=species_threshold,
                                                min_champion_size=5,
                                                balance_top=0.5,
                                                balance_bottom=0.25,
                                                balance_focus=balance_focus,
                                                stagnant_generations=20)

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
        self.specieator.add_all_to_species(self.population, self.generation_count)

    def repopulate(self, fitness: List[float]):
        self.generation_count += 1

        for i in range(len(fitness)):
            self.population[i].score = fitness[i]
        self.specieator.score_species(self.generation_count)
        self.specieator.remove_stagnant_species()

        SIS = self.specieator.get_species_champions()
        reproducers = int(self.population_size * self.reproducer_ratio) - len(SIS)

        map(self.population.remove, SIS)

        self.specieator.adjust_scores()

        self.population.sort(reverse=True)
        self.reprocucers = SIS + self.population[:reproducers]

        self.population = SIS

        self.specieator.remove_all_but(self.reprocucers)

        new_nets = self.breed_new_nets(self.population_size - len(SIS))

        self.specieator.remove_all_but(self.population)

        self.population.extend(new_nets)
        self.specieator.clean()

        if self.specieator.balance_threshold(self.population_size):
            self.specieator.clear()
            self.specieator.add_all_to_species(self.population, self.generation_count)
        else:
            self.specieator.add_all_to_species(new_nets, self.generation_count)

    def breed_new_nets(self, size: int):
        reprocucers = len(self.reprocucers)
        new_nets = []
        for i in range(size):
            if random.random() > self.asexual_breading_rate:
                if random.random() < self.inter_species_breeding_rate:
                    child = self.reprocucers[i % reprocucers].breed(random.choice(self.reprocucers[:reprocucers]))
                else:
                    specie = self.specieator.get_species(self.reprocucers[i % reprocucers])

                    child = self.reprocucers[i % reprocucers].breed(
                        random.choice(self.specieator.species[specie]))
            else:
                child = self.reprocucers[i % reprocucers].replicate()
            new_nets.append(child)
        return new_nets

    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, dot_size: int = 10):
        self.specieator.draw(screen, x, y, width, height, dot_size)