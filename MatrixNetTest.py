import MatrixNet, numpy, pygame, random

def imitater(ar):
    return [
        1.0 if ar[0] == ar[1] == 0.0 else 0,
        1.0 if ar[0] == 0.0 and ar[1] == 1.0 else 0,
        1.0 if ar[0] == 1.0 and ar[1] == 0.0 else 0,
        1.0 if ar[0] == ar[1] == 1.0 else 0
    ]


def relu(x):
    return 0.0 if x < 0 else x

relu_Array = numpy.vectorize(relu)

def redir(x):
    return 0.0 if x <= 0 else 1

relu_der_Array = numpy.vectorize(redir)

Net = MatrixNet.MatrixNet([2, 4, 4], [-1.0, 1.0])
NUMBER_OF_STATES = 4

# while numpy.linalg.norm(Net.getOut()

pygame.init()
Screen = pygame.display.set_mode([400, 400])
pygame.key.set_repeat(100, 50)
Screen.fill([0, 0, 100])

KEEP = True
States = []
for i in range(0, NUMBER_OF_STATES):
    States.append([random.choice([1, 0]), random.choice([1, 0]), random.choice([1, 0]), random.choice([1, 0])])
States = [[1.0, 1.0],
          [0.0, 1.0],
          [1.0, 0.0],
          [0.0, 0.0]]
StatesIndex = 0
Input = States[StatesIndex]
Net.setIn(Input)
Ratio = 1.0
found = False
delay = 1
while KEEP:
    pygame.time.delay(delay)
    # input("-----------------------")
    Net.getOutThreshold()
    Net.learnWithThreshold(Ratio, imitater(Input))
    # print(Input)
    # Net.learn(Ratio, Input)
    Screen.fill([0, 0, 100])
    Net.draw(Screen, 10, 10, 50)
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            KEEP = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
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
