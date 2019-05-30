import math

from EvolvingNet import EvolvingNet
import MatrixNet, numpy, pygame, random

STATE_SIZE = 6
NUMBER_OF_STATES = 6

IN_DEM = STATE_SIZE
OUT_DEM = STATE_SIZE

POPULATION_SIZE = 100
GENERATION_LENGTH = 100
MUTABILITY = 0.5
SURVIVOR_RATIO = 0.5

WIDTH = 1000
HEIGHT = 800
DISPLAY_ROW_SIZE = 5
DISPLAY_ROW_COUNT = 2
ERROR_SIZE = 500


def imitater(ar):
    return ar


pygame.init()

Screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.key.set_repeat(100, 50)
Screen.fill([0, 0, 100])

KEEP = True

delay = 0
error_array = []

while KEEP:
    pygame.time.delay(delay)
    if len(error_array) > ERROR_SIZE:
        error_array.pop(0)

    Screen.fill([0, 0, 100])

    for i in range(DISPLAY_ROW_COUNT * DISPLAY_ROW_SIZE):
        Net[1].draw(Screen, (i % DISPLAY_ROW_SIZE) * (WIDTH // DISPLAY_ROW_SIZE),
                    (i // DISPLAY_ROW_SIZE) * ((HEIGHT - 200) // DISPLAY_ROW_COUNT),
                    WIDTH // DISPLAY_ROW_SIZE,
                    (HEIGHT - 200) // DISPLAY_ROW_COUNT,
                    10)
    pygame.draw.line(Screen, [0, 0, 0], [0, HEIGHT - 200], [WIDTH, HEIGHT - 200])
    pygame.draw.line(Screen, [255, 255, 255], [0, HEIGHT - 100], [WIDTH, HEIGHT - 100])
    for i in range(len(error_array)):
        # print( [int(10 + (HEIGHT - 10) * (len(error_array) - i)), error_array[i]])
        pygame.draw.circle(Screen, [255, 0, 0],
                           [int(10 + (WIDTH - 10) * (len(error_array) - i) / len(error_array)), error_array[i]], 5)
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
                    delay = 500
                else:
                    delay = 1
                # print("changed")
    # if numpy.linalg.norm(Net.getOutThreshold() - numpy.reshape(numpy.array(Input), (1, 1))) < .1 and not found:
    #     print("Found")
    #     found = True
    count += 1
