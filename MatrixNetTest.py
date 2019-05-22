import math

import MatrixNet, numpy, pygame, random

STATE_SIZE = 6
NUMBER_OF_STATES = 6
WIDTH = 1000
HEIGHT = 800
ERROR_SIZE = 500
DIMEN = [STATE_SIZE, STATE_SIZE * 2, STATE_SIZE // 2 + 1]
Ratio = 1.0


def numtobits(num, bits):
    l = [0] * bits
    num = num % (2 ** bits)
    for i in range(bits - 1, -1, -1):
        l[i] = int(num / (2 ** i))
        num = num % (2 ** i)
    return l


def bitstonum(bits):
    num = 0
    for i in range(0, len(bits)):
        num += (2 ** i) * bits[i]
    return num


out = numtobits(5, 5)
print(out)
print(bitstonum(out))


def imitater(ar):
    n1 = bitstonum(ar[0:STATE_SIZE // 2])
    n2 = bitstonum(ar[STATE_SIZE // 2:])
    SUM = n1 + n2
    return numtobits(SUM, STATE_SIZE // 2 + 1)


# input = numtobits(5,5) + numtobits(3,5)
# print("output: " + str(bitstonum(imitater(input))))


tanh_Array = numpy.tanh


def tanh_derivative(x):
    return 1.0 - x ** 2.0


tanh_der_Array = numpy.vectorize(tanh_derivative)


def tanh_color_formula(x):
    return int((1.0 + x) * 127.5)


def relu(x):
    return 0.0 if x < 0 else x


relu_Array = numpy.vectorize(relu)


def relu_derivative(x):
    return 0.0 if x < 0.0 else 1.0


relu_der_Array = numpy.vectorize(relu_derivative)


def sigmoid(x):
    return 1.0 / (1.0 + math.e ** -float(x))


def relu_color_formula(x):
    return (int(x * 255.) if x * 255.0 <= 255.0 else 255.0) if x > 0.0 else 0.0


Net = MatrixNet.MatrixNet(DIMEN, [-1.0, 1.0])  # , relu_Array, relu_der_Array, relu_color_formula)

# while numpy.linalg.norm(Net.getOut()

pygame.init()
Screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.key.set_repeat(100, 50)
Screen.fill([0, 0, 100])

KEEP = True
States = []
for i in range(0, NUMBER_OF_STATES):
    State = []
    for i in range(0, STATE_SIZE):
        State.append(random.choice([1, 0]))
    States.append(State)
    # States.append([, random.choice([1, 0]), random.choice([1, 0]), random.choice([1, 0])])
States = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
          ]
StatesIndex = 0
Input = States[StatesIndex]
Net.setIn(Input)
found = False
delay = 0
error_array = []
while KEEP:
    pygame.time.delay(delay)
    # input("-----------------------")
    Net.getOutThreshold()
    error_array.append(int((HEIGHT - 100) - Net.learnWithThreshold(Ratio, imitater(Input)) * 50.0))
    if len(error_array) > ERROR_SIZE:
        error_array.pop(0)
    # print(Input)
    # Net.learn(Ratio, Input)
    Screen.fill([0, 0, 100])
    Net.draw(Screen, 10, 10, 100)
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

            elif event.key == pygame.K_1:
                NUMBER_OF_STATES = 6
                States = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                          ]
            elif event.key == pygame.K_2:
                NUMBER_OF_STATES = 12
                States = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],

                          [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                          ]
            elif event.key == pygame.K_3:
                NUMBER_OF_STATES = 18
                States = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],

                          [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],

                          [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                          [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                          [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                          ]
            elif event.key == pygame.K_4:
                NUMBER_OF_STATES = 24
                States = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],

                          [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],

                          [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                          [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                          [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],

                          [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                          [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                          [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                          ]
            elif event.key == pygame.K_5:
                NUMBER_OF_STATES = 30
                States = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],

                          [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],

                          [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                          [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                          [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],

                          [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                          [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                          [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],

                          [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                          [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                          [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                          [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                          [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                          ]
            elif event.key == pygame.K_s:
                if delay == 1:
                    delay = 500
                else:
                    delay = 1
            if event.key == pygame.K_DOWN:
                Ratio -= .1
                print(Ratio)
            elif event.key == pygame.K_UP:
                Ratio += .1
                print(Ratio)
            elif event.key == pygame.K_SPACE:
                Input = [random.choice([1, 0]), random.choice([1, 0]), random.choice([1, 0]), random.choice([1, 0])]
                Net.setIn(Input)
                found = False
                # print("changed")
    # if numpy.linalg.norm(Net.getOutThreshold() - numpy.reshape(numpy.array(Input), (1, 1))) < .1 and not found:
    #     print("Found")
    #     found = True
    StatesIndex = (StatesIndex + 1) % NUMBER_OF_STATES
    Input = States[StatesIndex]
    Net.setIn(Input)
    # KEEP = False

    # Input = [random.choice([1, 0]), random.choice([1, 0]), random.choice([1, 0]), random.choice([1, 0])]
    # Net.setIn(Input)
# Net.learn(.5, [[1], [1], [0], [1]])

# print "sucess"
# while  KEEP:
#
#     # Net.learn(Ratio, Input)
#     # Net.getOut()
#     Screen.fill([0, 0, 100])
#     Net.draw(Screen, 10, 10, 50)
#     pygame.display.flip()
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             KEEP = False
#         # elif event.type == pygame.KEYDOWN:
#         #     if event.key == pygame.K_DOWN:
#         #         Ratio -= .1
#         #         print Ratio
#         #     elif event.key == pygame.K_UP:
#         #         Ratio += .1
#         #         print Ratio
