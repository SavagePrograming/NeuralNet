import MatrixNet, numpy, pygame, random

Net = MatrixNet.MatrixNet([4, 4, 4], [0, 0])
NUMBER_OF_STATES = 2

# while numpy.linalg.norm(Net.getOut()

pygame.init()
Screen = pygame.display.set_mode([400,400])
pygame.key.set_repeat(100, 50)
Screen.fill([0, 0, 100])

KEEP = True
States = []
for i in range(0, NUMBER_OF_STATES):
    States.append([random.choice([1, 0]), random.choice([1,0]), random.choice([1,0]), random.choice([1,0])])
States = [[1,1,0,0], [1,1,1,1]]
StatesIndex = 0
Input = States[StatesIndex]
Net.setIn(Input)
Ratio = 5.
found = False
while KEEP:
    raw_input("-----------------------")
    Net.getOutThreshold()
    Net.learnWithThreshold(Ratio, Input)
    Screen.fill([0, 0, 100])
    Net.draw(Screen, 10, 10, 50)
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            KEEP = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                Ratio -= .1
                # print Ratio
            elif event.key == pygame.K_UP:
                Ratio += .1
                # print Ratio
            elif event.key == pygame.K_SPACE:
                Input = [random.choice([1, 0]), random.choice([1, 0]), random.choice([1, 0]), random.choice([1, 0])]
                Net.setIn(Input)
                found = False
                print "changed"
    if numpy.linalg.norm(Net.getOutThreshold()- numpy.reshape(numpy.array(Input), (4,1)) ) < .1 and not found:
        print "Found"
        # found = True
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