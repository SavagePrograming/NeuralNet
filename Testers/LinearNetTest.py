import numpy, pygame, random
from Nets import LinearNet

STATE_SIZE = 2
NUMBER_OF_STATES = 100
WIDTH = 1000
HEIGHT = 800
ERROR_SIZE = 500
DOT_SIZE = 10
DIMEN = [STATE_SIZE, 0, 1]
Ratio = 5.0


def imitater(ar):
    return [1.0] if (ar[0] == 1.0) != (ar[1] == 1.0) else [0.0]


enabled_weights = ([[True] * sum(DIMEN[1:-1]) + [False]] * DIMEN[0]) + [[True] * (sum(DIMEN[1:-1]) + 1)] + ([[False] * sum(DIMEN[1:-1]) + [True]] *sum(DIMEN[1:-1]))

pygame.init()
Screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.key.set_repeat(100, 50)
Screen.fill([150, 150, 150])

Net = LinearNet.LinearNet(in_dem=DIMEN[0],
                          out_dem=DIMEN[-1],
                          middle_dem=sum(DIMEN[1:-1]),
                          weight_range=[1.0, -1.0],
                          enabled_weights=enabled_weights)
Net.update(screen=Screen,
           x=0,
           y=0,
           width=WIDTH,
           height=HEIGHT - 200,
           scale_dot=DOT_SIZE)
KEEP = True
States = []
for i in range(0, NUMBER_OF_STATES):
    State = []
    for i in range(0, STATE_SIZE):
        State.append(random.choice([1, 0]))
    States.append(State)

StatesIndex = 0
Input = States[StatesIndex]
Net.set_in(Input)
found = False
delay = 0
error_array = []
while KEEP:
    pygame.time.delay(delay)

    Net.get_out()

    error = Net.learn(Ratio, imitater(Input))
    error_array.append(int((HEIGHT - 100) - error * 50.0))
    if len(error_array) > ERROR_SIZE:
        error_array.pop(0)
    print(error)
    Screen.fill([150, 150, 150])
    Net.draw()

    pygame.draw.line(Screen, [0, 0, 0], [0, HEIGHT - 200], [WIDTH, HEIGHT - 200])
    pygame.draw.line(Screen, [255, 255, 255], [0, HEIGHT - 100], [WIDTH, HEIGHT - 100])
    for i in range(len(error_array)):
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
            if event.key == pygame.K_s:
                if delay == 0:
                    delay = 500
                else:
                    delay = 0

    StatesIndex = (StatesIndex + 1) % NUMBER_OF_STATES
    Input = States[StatesIndex]
    Net.set_in(Input)
