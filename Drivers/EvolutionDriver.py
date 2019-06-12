import random
from typing import ClassVar, List

from Drivers.Driver import Driver
from Nets.EvolvingNet import EvolvingNet
from Simulations import EvolutionSimulation


class EvolutionDriver(Driver):
    def __init__(self,
                 population_size: int,
                 survivor_ratio: float,
                 simulation: EvolutionSimulation,
                 generation_size: int,
                 row_size: int,
                 row_count: int,
                 mutability: float = 0.5,
                 evolving_class: ClassVar = EvolvingNet):
        super(EvolutionDriver, self).__init__(
            population_size=population_size,
            simulation=simulation,
            generation_size=generation_size,
            row_size=row_size,
            row_count=row_count,
            mutability=mutability,
            evolving_class=evolving_class
        )
        self.survivor_ratio: float = survivor_ratio

    def repopulate(self, fitness: List[float]):
        for i in range(len(fitness)):
            self.population[i].Score = fitness[i]
        self.population.sort(reverse=True)
        self.population = self.population[:int(self.population_size * self.survivor_ratio)]

        size = len(self.population)
        for i in range(self.population_size - len(self.population)):
            self.population.append(self.population[i % size].breed(random.choice(self.population[:size])))
