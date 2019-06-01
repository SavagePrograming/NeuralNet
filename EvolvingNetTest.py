import math

import EvolutionDriver, EvolutionSpeciationDriver
import EvolutionSimulation
from EvolvingNet import EvolvingNet
from StaticEvolvingNet import StaticEvolvingNet
import MatrixNet, numpy, pygame, random

STATE_SIZE = 2
NUMBER_OF_STATES = 6

IN_DEM = STATE_SIZE
OUT_DEM = 1

POPULATION_SIZE = 100
GENERATION_LENGTH = 4
MUTABILITY = 1.0
SURVIVOR_RATIO = 0.25
SPECIES_THRESHOLD = 6.0

WIDTH = 1000
HEIGHT = 800
DISPLAY_ROW_SIZE = 1
DISPLAY_ROW_COUNT = 1
ERROR_SIZE = 500


def imitater(ar):
    return 1.0 if ar[0] != ar[1] else 0.0


sim = EvolutionSimulation.EvolveSimulation(imitater, IN_DEM, OUT_DEM)

driver = EvolutionSpeciationDriver.EvolutionSpeciationDriver(POPULATION_SIZE, SURVIVOR_RATIO, sim, GENERATION_LENGTH, SPECIES_THRESHOLD, MUTABILITY, evolving_class=StaticEvolvingNet)

pygame.init()

Screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.key.set_repeat(100, 50)
Screen.fill([0, 0, 0])

KEEP = True
delay = 0
testing = False

error_array = []

while KEEP:
    pygame.time.delay(delay)

    # Screen.fill([0, 0, 100])
    if testing:
        driver.test()
        driver.draw(Screen, DISPLAY_ROW_SIZE, DISPLAY_ROW_COUNT, 0, 0, WIDTH, HEIGHT - 200, 10)
    else:
        driver.run_visual(Screen, DISPLAY_ROW_SIZE, DISPLAY_ROW_COUNT, 0, 0, WIDTH, HEIGHT - 200, 10)
    error_array.append((driver.Maximum, driver.Average, driver.Median, driver.Minimum))
    if len(error_array) > ERROR_SIZE:
        error_array.pop(0)



    Screen.fill([0, 0, 0])
    driver.draw(Screen, DISPLAY_ROW_SIZE, DISPLAY_ROW_COUNT, 0, 0, WIDTH, HEIGHT - 200, 10)
    pygame.draw.line(Screen, [255, 255, 255], [0, HEIGHT - 200], [WIDTH, HEIGHT - 200])
    if len(error_array) > 2:
        print(error_array[-2][0])
        print(error_array[-1][0])
        if error_array[-2][0] > error_array[-1][0]:
            print("Down")
    for i in range(len(error_array)):
        pygame.draw.circle(Screen, [255, 0, 0],
                           [int(10 + (WIDTH - 10) * (len(error_array) - i) / len(error_array)),
                            int(HEIGHT - 100 * error_array[i][0])], 5)
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            KEEP = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_0:
                NUMBER_OF_STATES = 20
                States = []
                for i in range(0, NUMBER_OF_STATES):
                    State = []
                    for i in range(0, STATE_SIZE):
                        State.append(random.choice([1, 0]))
                    States.append(State)
            elif event.key == pygame.K_s:
                if delay == 1:
                    delay = 1000
                else:
                    delay = 1
            elif event.key == pygame.K_SPACE:
                testing = not testing
