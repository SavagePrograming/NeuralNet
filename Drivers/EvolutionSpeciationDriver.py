import random
from typing import ClassVar, List

import pygame
from math import ceil
from numpy import mean, median, math

from Drivers.Driver import Driver
from Nets.EvolvingNet import EvolvingNet
from Simulations.EvolutionSimulation import EvolutionSimulation


def clean(a):
    i = 0
    while i < len(a):
        if not a[i]:
            del a[i]
        else:
            i += 1


class EvolutionSpeciationDriver(Driver):
    def __init__(self,
                 population_size: int,
                 survivor_ratio: float,
                 simulation: EvolutionSimulation,
                 generation_size: int,
                 species_threshold: float,
                 species_independent_survivor_ratio: float,
                 balancing_focus: float,
                 mutability: float = 0.5,
                 evolving_class: ClassVar = EvolvingNet):
        super(EvolutionSpeciationDriver, self).__init__(
            population_size=population_size,
            simulation=simulation,
            generation_size=generation_size,
            row_size=0,
            row_count=0,
            mutability=mutability,
            evolving_class=evolving_class
        )
        self.balance_top: float = .5
        self.balance_bottom: float = .25
        self.species_threshold: float = species_threshold
        self.balance_focus: float = balancing_focus
        self.SISR: float = species_independent_survivor_ratio

        self.survivor_ratio: float = survivor_ratio
        self.species: List[List[evolving_class]] = []
        for child in self.population:
            self.add_to_specie(child)

    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, dot_size: int = 10):

        self.row_count = int(math.sqrt(len(self.species)))
        self.row_size = math.ceil(len(self.species) / self.row_count)
        super(EvolutionSpeciationDriver, self).draw(screen, x, y, width, height, dot_size)

    def repopulate(self, fitness: List[float]):
        survivors = int(self.population_size * self.survivor_ratio) - ceil(self.population_size * self.SISR)
        for i in range(len(fitness)):
            self.population[i].Score = fitness[i]
        SIS = []
        if self.SISR < .25:
            for i in range(ceil(self.population_size * self.SISR)):
                MAX = max(self.population)
                self.population.remove(MAX)
                SIS.append(MAX)
        else:
            self.population.sort(reverse=True)
            SIS = self.population[:ceil(self.population_size * self.SISR)]
            self.population = self.population[ceil(self.population_size * self.SISR):]

        for s in range(len(self.species)):
            size = len(self.species[s])
            for net in self.species[s]:
                net.score /= size
        self.population.sort(reverse=True)
        self.population = self.population[:survivors]
        self.population = SIS + self.population
        for s in range(len(self.species)):
            n = 0
            while n < len(self.species[s]):
                if self.species[s][n] not in self.population:
                    del self.species[s][n]
                else:
                    n += 1
        if self.balance_focus == 0:
            clean(self.species)
            self.speciate(self.population_size - int(self.population_size * self.survivor_ratio))
        else:
            print("Species:" + str(len(self.species)))
            if len(self.species) / self.population_size > self.balance_top:
                self.species_threshold += self.balance_focus
                clean(self.species)
                self.respeciate(self.population_size - int(self.population_size * self.survivor_ratio))
            elif len(self.species) / self.population_size < self.balance_bottom:
                self.species_threshold -= self.balance_focus
                clean(self.species)
                self.respeciate(self.population_size - int(self.population_size * self.survivor_ratio))
            else:
                clean(self.species)
                self.speciate(self.population_size - int(self.population_size * self.survivor_ratio))

    def respeciate(self, size: int):
        survivors = len(self.population)
        self.species = []
        for i in range(size):
            if random.random() < .5:
                child = self.population[i % survivors].breed(random.choice(self.population[:survivors]))
            else:
                child = self.population[i % survivors].replicate()
            self.population.append(child)
        for child in self.population:
            self.add_to_specie(child)
        print("Threshold: " + str(self.species_threshold))
        print("Species:" + str(len(self.species)))

    def speciate(self, size: int):
        survivors = len(self.population)
        for i in range(size):
            if random.random() < .5:
                child = self.population[i % survivors].breed(random.choice(self.population[:survivors]))
            else:
                child = self.population[i % survivors].replicate()
            self.population.append(child)
            self.add_to_specie(child)
        print("Species:" + str(len(self.species)))

    def add_to_specie(self, child):
        remaining = True
        print(child)
        print(self.species)
        for s in range(len(self.species)):
            if child.distance(self.species[s][0]) < self.species_threshold:
                self.species[s].append(child)
                # print(self.species[s])
                remaining = False
                break
        if remaining:
            self.species.append([child])
        print(self.species)