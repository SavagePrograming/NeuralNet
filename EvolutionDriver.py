import random

from EvolvingNet import EvolvingNet


class EvolutionDriver:
    def __init__(self, population_size, survivor_ratio, simlulation, generation_size, mutability=0.5,
                 evolving_class=EvolvingNet):
        self.InDem = simlulation.InDem
        self.OutDem = simlulation.OutDem
        self.PopulationSize = population_size
        self.Mutability = mutability
        self.SurvivorRatio = survivor_ratio
        self.Simulation = simlulation
        self.EvolvingClass = evolving_class
        self.GenerationSize = generation_size

        self.Population = []
        for i in range(self.PopulationSize):
            self.Population.append((0.0, self.EvolvingClass(self.InDem, self.OutDem, [], [], mutability=mutability)))

    def run(self):
        Error = self.Simulation.run_generations(self.Population, self.GenerationSize)
        self.repopulate(Error)

    def draw(self, screen, row_size, row_count, x, y, width, height):
        for i in range(row_count * row_size):
            self.Population[i].draw(screen,
                                    x + (i % row_size) * (width // row_size),
                                    y + (i // row_size) * ((height) // row_count),
                                    width // row_size,
                                    (height) // row_count,
                                    10)

    def repopulate(self, Error):
        Sorter = []
        for i in range(len(Error)):
            Sorter.append((Error[i], self.Population[i]))
        Sorter.sort()
        NewPopulation = []
        for i in range(int(self.PopulationSize * self.SurvivorRatio)):
            for i in range(len(Sorter)):
                if self.SurvivorRatio > random.random():
                    NewPopulation.append(Sorter.pop(i))
                    break
                elif i == len(Sorter) - 1:
                    NewPopulation.append(Sorter.pop(i))
                    break
        NewPopulation.sort()
        for i in range(len(NewPopulation)):
            NewPopulation[i] = NewPopulation[i][1]
        size = len(NewPopulation)
        for i in range(self.PopulationSize - len(NewPopulation)):
            NewPopulation.append(NewPopulation[i % size].breed(random.choice(NewPopulation[:size])))
        self.Population = NewPopulation
