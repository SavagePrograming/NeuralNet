import random

from numpy import mean, median

from Nets.EvolvingNet import EvolvingNet


class EvolutionDriver:
    def __init__(self, population_size, survivor_ratio, simlulation, generation_size, row_size, row_count, mutability=0.5,
                 evolving_class=EvolvingNet):
        self.InDem = simlulation.InDem
        self.OutDem = simlulation.OutDem
        self.row_size = row_size
        self.row_count = row_count
        self.PopulationSize = population_size
        self.Mutability = mutability
        self.SurvivorRatio = survivor_ratio
        self.Simulation = simlulation
        self.EvolvingClass = evolving_class
        self.GenerationSize = generation_size
        self.Median = 0.0
        self.Average = 0.0
        self.Maximum = 0.0
        self.Minimum = 0.0

        self.Population = []
        for i in range(self.PopulationSize):
            self.Population.append(self.EvolvingClass(self.InDem, self.OutDem, 1, mutability=mutability))

    def run(self):
        self.Simulation.restart()
        Fitness = self.Simulation.run_generations(self.Population, self.GenerationSize)
        self.Average = mean(Fitness)
        self.Median = median(Fitness)
        self.Maximum = max(Fitness)
        self.Minimum = min(Fitness)
        self.repopulate(Fitness)

    def test(self):
        Fitness = self.Simulation.run(self.Population)
        self.Average = mean(Fitness)
        self.Median = median(Fitness)
        self.Maximum = max(Fitness)
        self.Minimum = min(Fitness)

    def draw(self, screen, x, y, width, height, dot_size=10):
        for i in range(self.row_count * self.row_size):
            self.Population[i].draw(screen,
                                    x + (i % self.row_size) * (width // self.row_size),
                                    y + (i // self.row_size) * (height // self.row_count),
                                    width // self.row_size,
                                    height // self.row_count,
                                    dot_size)

    def run_visual(self, screen, x, y, width, height, dot_size=10):
        self.Simulation.restart()
        Fitness = self.Simulation.run_generations_visual(self.Population,
                                                         self.GenerationSize, self,
                                                         screen, x, y, width, height, dot_size)
        self.Average = mean(Fitness)
        self.Median = median(Fitness)
        self.Maximum = max(Fitness)
        self.Minimum = min(Fitness)
        self.repopulate(Fitness)


    def repopulate(self, Fitness):
        for i in range(len(Fitness)):
            self.Population[i].Score = Fitness[i]
        self.Population.sort(reverse=True)
        self.Population = self.Population[:int(self.PopulationSize * self.SurvivorRatio)]

        size = len(self.Population)
        for i in range(self.PopulationSize - len(self.Population)):
            self.Population.append(self.Population[i % size].breed(random.choice(self.Population[:size])))
