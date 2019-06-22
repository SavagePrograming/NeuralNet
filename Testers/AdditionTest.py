from Simulations import EvolutionDivisibleSimulation
from Drivers import EvolutionSpeciationDriver
from Nets.StaticEvolvingNet import StaticEvolvingNet
import pygame, random


POPULATION_SIZE = 200
GENERATION_LENGTH = 100
MUTABILITY = 50.0
SURVIVOR_RATIO = 0.25
SISR = .02
SPECIES_THRESHOLD = 110.0
BALANCE_FOCUS = 1.0
LAYERS = 0


WIDTH = 1000
HEIGHT = 800
DISPLAY_ROW_SIZE = 1
DISPLAY_ROW_COUNT = 1
ERROR_SIZE = 500

sim = EvolutionDivisibleSimulation.DivisionSimulation(LAYERS)

driver = EvolutionSpeciationDriver.EvolutionSpeciationDriver(POPULATION_SIZE, SURVIVOR_RATIO, sim,
                                                             GENERATION_LENGTH, SPECIES_THRESHOLD, SISR,
                                                             BALANCE_FOCUS, mutability=MUTABILITY,
                                                             evolving_class=StaticEvolvingNet)

pygame.init()

Screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.key.set_repeat(100, 50)

Screen.fill([150, 150, 150])

KEEP = True
delay = 0
testing = False

error_array = []

while KEEP:
    pygame.time.delay(delay)

    if testing:
        driver.test()
        driver.draw(Screen, DISPLAY_ROW_SIZE, DISPLAY_ROW_COUNT, 0, 0, WIDTH, HEIGHT - 200, 10)
    else:
        driver.run_visual(Screen, DISPLAY_ROW_SIZE, DISPLAY_ROW_COUNT, 0, 0, WIDTH, HEIGHT - 200, 2)
    error_array.append((driver.maximum, driver.average, driver.median, driver.minimum))
    if len(error_array) > ERROR_SIZE:
        error_array.pop(0)



    Screen.fill([150, 150, 150])
    driver.draw(Screen, DISPLAY_ROW_SIZE, DISPLAY_ROW_COUNT, 0, 0, WIDTH, HEIGHT - 200, 2)
    pygame.draw.line(Screen, [0, 0, 0], [0, HEIGHT], [WIDTH, HEIGHT], 5)
    pygame.draw.line(Screen, [200, 200, 200], [0, HEIGHT - 50], [WIDTH, HEIGHT - 50], 5)
    pygame.draw.line(Screen, [255, 255, 255], [0, HEIGHT - 100], [WIDTH, HEIGHT - 100], 5)
    pygame.draw.line(Screen, [200, 200, 200], [0, HEIGHT - 150], [WIDTH, HEIGHT - 150], 5)
    pygame.draw.line(Screen, [0, 0, 0], [0, HEIGHT - 200], [WIDTH, HEIGHT - 200], 5)
    # print(error_array[-1][0])
    for i in range(len(error_array)):
        pygame.draw.circle(Screen, [255, 0, 0],
                           [int(10 + (WIDTH - 10) * (i) / len(error_array)),
                            int(HEIGHT - 50 * error_array[i][3])], 5)
        pygame.draw.circle(Screen, [0, 255, 0],
                           [int(10 + (WIDTH - 10) * (i) / len(error_array)),
                            int(HEIGHT - 50 * error_array[i][2])], 5)
        pygame.draw.circle(Screen, [0, 0, 255],
                           [int(10 + (WIDTH - 10) * (i) / len(error_array)),
                            int(HEIGHT - 50 * error_array[i][1])], 5)
        pygame.draw.circle(Screen, [0, 0, 0],
                           [int(10 + (WIDTH - 10) * (i) / len(error_array)),
                            int(HEIGHT - 50 * error_array[i][0])], 5)
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            KEEP = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_0:
                NUMBER_OF_STATES = 20
                States = []
                for i in range(NUMBER_OF_STATES):
                    State = []
                    for i in range(STATE_SIZE):
                        State.append(random.choice([1, 0]))
                    States.append(State)
            elif event.key == pygame.K_s:
                if delay == 1:
                    delay = 1000
                else:
                    delay = 1
            elif event.key == pygame.K_SPACE:
                testing = not testing
