import random
from typing import ClassVar, List

import pygame
from math import ceil
from numpy import mean, median, math

from Nets.EvolvingNet import EvolvingNet
from Simulations.EvolutionSimulation import EvolutionSimulation


def clean(a):
    i = 0
    while i < len(a):
        if not a[i]:
            del a[i]
        else:
            i += 1


class EvolutionSpeciationDriver:
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
        self.balance_top: float = .5
        self.balance_bottom: float = .25
        self.species_threshold: float = species_threshold
        self.balance_focus: float = balancing_focus
        self.SISR: float = species_independent_survivor_ratio
        self.in_dem: int = simulation.InDem
        self.out_dem: int = simulation.OutDem
        self.population_size: int = population_size
        self.mutability: float = mutability
        self.survivor_ratio: float = survivor_ratio
        self.simulation: EvolutionSimulation = simulation
        self.evolving_class = evolving_class
        self.generation_size: int = generation_size
        self.median: float = 0.0
        self.average: float = 0.0
        self.maximum: float = 0.0
        self.minimum: float = 0.0
        self.species: List[List[evolving_class]] = []

        self.population: List[evolving_class] = []
        for i in range(self.population_size):
            child = self.evolving_class(self.in_dem, self.out_dem, self.simulation.Layers, mutability=mutability)
            self.population.append(child)
            self.add_to_specie(child)

    def run(self):
        self.simulation.restart()
        fitness = self.simulation.run_generations(self.population, self.generation_size)
        self.average = mean(fitness)
        self.median = median(fitness)
        self.maximum = max(fitness)
        self.minimum = min(fitness)
        self.repopulate(fitness)

    def test(self):
        fitness = self.simulation.run(self.population)
        self.average = mean(fitness)
        self.median = median(fitness)
        self.maximum = max(fitness)
        self.minimum = min(fitness)

    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, dot_size: int = 10):
        row_count = int(math.sqrt(len(self.species)))
        row_size = math.ceil(len(self.species) / row_count)
        for i in range(row_count * row_size):
            if len(self.species) > i:
                self.species[i][0].draw(screen,
                                        x + (i % row_size) * (width // row_size),
                                        y + (i // row_size) * (height // row_count),
                                        width // row_size,
                                        height // row_count,
                                        dot_size)

    def run_visual(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, dot_size: int = 10):
        self.simulation.restart()
        fitness = self.simulation.run_generations_visual(self.population, self.generation_size, self, screen, x, y,
                                                         width, height, dot_size)
        self.average = mean(fitness)
        self.median = median(fitness)
        self.maximum = max(fitness)
        self.minimum = min(fitness)
        self.repopulate(fitness)

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
        for s in range(len(self.species)):
            if child.distance(self.species[s][0]) < self.species_threshold:
                self.species[s].append(child)
                remaining = False
                break
        if remaining:
            self.species.append([child])
