import random

PROBABILITY = .5
SIZE = 50
TESTS = 1000
def test_one(prob):
    return 1 if random.random() < prob else 0
def discrete_tests(size, prob):
    return sum(map(test_one, [prob] * size))

def normal(size, prob):
    return int(size * numpy.random.normal(prob, prob)


buckets = [[0]*SIZE, [0]*SIZE]