import random

import pygame
from numpy import mean, median
from typing import ClassVar, List
from Nets.EvolvingNet import EvolvingNet
from Simulations import EvolutionSimulation


class EvolutionDriver:
    def __init__(self,
                 population_size: int,
                 survivor_ratio: float,
                 simulation: EvolutionSimulation,
                 generation_size: int,
                 row_size: int,
                 row_count: int,
                 mutability: float = 0.5,
                 evolving_class: ClassVar = EvolvingNet):
        self.in_dem: int = simulation.InDem
        self.out_dem: int = simulation.OutDem
        self.row_size: int = row_size
        self.row_count: int = row_count
        self.population_size: int = population_size
        self.mutability: float = mutability
        self.survivor_ratio: float = survivor_ratio
        self.simulation = simulation
        self.evolving_class = evolving_class
        self.generation_size: int = generation_size
        self.median: float = 0.0
        self.average: float = 0.0
        self.maximum: float = 0.0
        self.minimum: float = 0.0

        self.population: List[evolving_class] = []
        for i in range(self.population_size):
            self.population.append(self.evolving_class(self.in_dem, self.out_dem, 1, mutability=mutability))

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

    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, dot_size: int=10):
        for i in range(self.row_count * self.row_size):
            self.population[i].draw(screen,
                                    x + (i % self.row_size) * (width // self.row_size),
                                    y + (i // self.row_size) * (height // self.row_count),
                                    width // self.row_size,
                                    height // self.row_count,
                                    dot_size)

    def run_visual(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, dot_size: int=10):
        self.simulation.restart()
        fitness = self.simulation.run_generations_visual(self.population,
                                                         self.generation_size, self,
                                                         screen, x, y, width, height, dot_size)
        self.average = mean(fitness)
        self.median = median(fitness)
        self.maximum = max(fitness)
        self.minimum = min(fitness)
        self.repopulate(fitness)

    def repopulate(self, fitness: List[float]):
        for i in range(len(fitness)):
            self.population[i].Score = fitness[i]
        self.population.sort(reverse=True)
        self.population = self.population[:int(self.population_size * self.survivor_ratio)]

        size = len(self.population)
        for i in range(self.population_size - len(self.population)):
            self.population.append(self.population[i % size].breed(random.choice(self.population[:size])))
