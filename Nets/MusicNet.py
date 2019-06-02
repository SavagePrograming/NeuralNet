import numpy, pygame, random
from Nets import MatrixNet
from formulas import distance_formula

Net = MatrixNet.MatrixNet([4, 4, 4], [0., 0.])
NUMBER_OF_STATES = 3

# while numpy.linalg.norm(Net.getOut()

pygame.init()
Screen = pygame.display.set_mode([400, 400])
pygame.key.set_repeat(100, 50)
Screen.fill([0, 0, 100])

def drawPoint(x, y, scalex, scaley, distance):
    pygame.draw.circle(Screen, [int(255. / (1. + distance)), int(255. / (1. + distance)), int(255. / (1. + distance))],[int(x * scalex), 400 - int(y * scaley)], 2)

MAX = 1
KEEP = True
States = [[1,1,1,1],[1,1,0,0], [0,0,0,0]]
# for i in range(0, NUMBER_OF_STATES):
#     States.append([random.choice([1, 0]), random.choice([1,0]), random.choice([1,0]), random.choice([1,0])])
StatesIndex = 0
Input = States[StatesIndex]
Net.setIn(Input)
Ratio = 0.1
found = False
Points = []
while KEEP:

    Net.learn(Ratio, Input)
    Net.get_out()
    Screen.fill([0, 0, 100])
    Net.draw(Screen, 10, 10, 50)
    if Points:
        d = distance_formula(Net.get_out(), numpy.reshape(numpy.array(Input), (4,1)))
        MAX = max(MAX, max(Points))
        map(drawPoint, range(0, len(Points)), Points, [400. / len(Points)] * len(Points),
            [200. / float(MAX)] * len(Points), [d] * len(Points))

    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            KEEP = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                Ratio -= .1
            elif event.key == pygame.K_UP:
                Ratio += .1
            elif event.key == pygame.K_SPACE:
                Input = [random.choice([1, 0]), random.choice([1, 0]), random.choice([1, 0]), random.choice([1, 0])]
                Net.setIn(Input)
                found = False
    Points.append(distance_formula(Net.get_out(), numpy.reshape(numpy.array(Input), (4,1))))
    if len(Points) > 200:
        Points.pop(0)
    StatesIndex = (StatesIndex + 1) % NUMBER_OF_STATES
    Input = States[StatesIndex]
    Net.setIn(Input)
