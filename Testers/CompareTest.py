import numpy, pygame, random
from Nets import LinearNet, MatrixNet

STATE_SIZE = 2
NUMBER_OF_STATES = 4
WIDTH = 1000
HEIGHT = 800
ERROR_SIZE = 500
DIMEN = [STATE_SIZE, STATE_SIZE, 1]
WEIGHT_RATIO = [2.0, -2.0]
Ratio = 4.0


def imitater(ar):
    return [1.0] if (ar[0] == 1) != (ar[1] == 1) else [0.0]


enabled_weights = [[True, True, False],
                   [True, True, False],
                   [True, True, True],
                   [False, False, True],
                   [False, False, True]]
# weights = [[-.5, .5, 0],
#                    [.5, .5, 0],
#                    [.5, -.5, .5],
#                    [0, 0, .5],
#                    [0, 0, .5]]
# weight2 = [numpy.array([[ weights[0][0],  weights[1][0],   weights[2][0]],
#        [ weights[0][1],  weights[1][1],   weights[2][1]]]),
#            numpy.array([[weights[3][2],  weights[4][2],   weights[2][2]]])]
# print(weight2)
# [[True, True, True],
#                    [True, True, True],
#                    [True, True, True],
#                    [False, True, True],
#                    [False, False, True]]

# [[True, True, True, True, True, True],
#    [True, True, True, True, True, True],
#    [True, True, True, True, True, True],
#    [False, True, True, True, True, True],
#    [False, False, True, True, True, True],
#    [False, False, False, True, True, True],
#    [False, False, False, False, True, True],
#    [False, False, False, False, False, True]]
limit = 0
Net = LinearNet.LinearNet(in_dem=DIMEN[0], out_dem=DIMEN[-1], middle_dem=sum(DIMEN[1:-1]),
                          weight_range=WEIGHT_RATIO)#, enabled_weights=enabled_weights)
Net2 = MatrixNet.MatrixNet(Dem= DIMEN, weight_range=WEIGHT_RATIO)
# Net2.WeightArray = weight2
# while numpy.linalg.norm(Net.getOut()

pygame.init()
Screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.key.set_repeat(100, 50)
Screen.fill([150, 150, 150])

KEEP = True
States = []
for i in range(0, NUMBER_OF_STATES):
    State = []
    for i in range(0, STATE_SIZE):
        State.append(random.choice([1, 0]))
    States.append(State)
    # States.append([, random.choice([1, 0]), random.choice([1, 0]), random.choice([1, 0])])
States = [[0.0, 1.0],
          [1.0, 1.0],
          [1.0, 0.0],
          [0.0, 0.0]]
StatesIndex = 0
Input = States[StatesIndex]
Net.set_in(Input)
Net2.set_in(Input)
found = False
delay = 0
error_array = []
error_array2 = []
while KEEP:
    if limit != 0 and limit < StatesIndex:
        KEEP = False
    pygame.time.delay(delay)
    # input("-----------------------")
    Net.get_out()
    Net2.get_out()
    # print(imitater(Input))
    error = Net.learn(Ratio, imitater(Input))
    error_array.append(int((HEIGHT - 100) - error * 50.0))
    error = Net2.learn(Ratio, imitater(Input))
    error_array2.append(int((HEIGHT - 100) - error * 50.0))
    # print("=================================")
    # print(Net.weights)
    # print(Net.node_back)
    # print("            ")
    # print(Net2.WeightArray[0])
    # print(Net2.WeightArray[1])
    if len(error_array) > ERROR_SIZE:
        error_array.pop(0)
        error_array2.pop(0)

    # print(error)
    Screen.fill([150, 150, 150])
    Net.draw(Screen, 0, 0, WIDTH / 2, HEIGHT - 200, 10)
    Net2.draw(Screen, WIDTH / 2, 0, WIDTH / 2, HEIGHT - 200, 10)

    pygame.draw.line(Screen, [0, 0, 0], [0, HEIGHT - 200], [WIDTH, HEIGHT - 200])
    pygame.draw.line(Screen, [50, 50, 50], [0, HEIGHT - 100], [WIDTH, HEIGHT - 100])
    for i in range(len(error_array)):
        pygame.draw.circle(Screen, [255, 0, 0],
                           [int(10 + (WIDTH - 10) * (len(error_array) - i) / len(error_array)), error_array[i]], 5)

        pygame.draw.circle(Screen, [150, 0, 200],
                           [int(10 + (WIDTH - 10) * (len(error_array) - i) / len(error_array)), error_array2[i]], 5)

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

    StatesIndex += 1
    Input = States[StatesIndex % NUMBER_OF_STATES]
    Net.set_in(Input)
    Net2.set_in(Input)
