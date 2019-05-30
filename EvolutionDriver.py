import random

from numpy import mean, median

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
        self.Median = 0.0
        self.Average = 0.0
        self.Maximum = 0.0
        self.Minimum = 0.0

        self.Population = []
        for i in range(self.PopulationSize):
            weights = [[]]
            for i in range(random.randint(0,1)):
                weights[0].append([])
            weights[0].append([(random.randint(0,2), random.random() * 2.0 - 1.0)])
            self.Population.append(self.EvolvingClass(self.InDem, self.OutDem, [], weights, mutability=mutability))

    def run(self):
        self.Simulation.restart()
        Error = self.Simulation.run_generations(self.Population, self.GenerationSize)
        self.Average = mean(Error)
        self.Median = median(Error)
        self.Maximum = max(Error)
        self.Minimum = min(Error)
        self.repopulate(Error)

    def draw(self, screen, row_size, row_count, x, y, width, height, dot_size=10):
        for i in range(row_count * row_size):
            self.Population[i].draw(screen,
                                    x + (i % row_size) * (width // row_size),
                                    y + (i // row_size) * (height // row_count),
                                    width // row_size,
                                    height // row_count,
                                    dot_size)

    def run_visual(self, screen, row_size, row_count, x, y, width, height, dot_size=10):
        self.Simulation.restart()
        Error = self.Simulation.run_generations_visual(self.Population, self.GenerationSize, self, screen, row_size,
                                                       row_count, x, y, width, height, dot_size)
        self.Average = mean(Error)
        self.Median = median(Error)
        self.Maximum = max(Error)
        self.Minimum = min(Error)
        self.repopulate(Error)

    def draw(self, screen, row_size, row_count, x, y, width, height, dot_size=10):
        for i in range(row_count * row_size):
            self.Population[i].draw(screen,
                                    x + (i % row_size) * (width // row_size),
                                    y + (i // row_size) * (height // row_count),
                                    width // row_size,
                                    height // row_count,
                                    dot_size)

    def repopulate(self, Error):
        sorter = []
        for i in range(len(Error)):
            sorter.append((Error[i], self.Population[i]))
        sorter.sort()
        new_population = []
        for i in range(int(self.PopulationSize * self.SurvivorRatio)):
            for j in range(len(sorter)):
                if self.SurvivorRatio > random.random():
                    new_population.append(sorter.pop(j))
                    break
                elif j == len(sorter) - 1:
                    new_population.append(sorter.pop(j))
                    break
        new_population.sort()
        # print(new_population[0])
        # print(new_population[499])
        for i in range(len(new_population)):
            new_population[i] = new_population[i][1]
        size = len(new_population)
        for i in range(self.PopulationSize - len(new_population)):
            new_population.append(new_population[i % size].breed(random.choice(new_population[:size])))
        self.Population = new_population
