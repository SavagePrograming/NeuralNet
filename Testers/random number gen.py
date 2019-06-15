import random

import math
import numpy

PROBABILITY = .5
SIZE = 10
TESTS = 100000

def normal(size, prob):

    return min(size, max(0, int(size * numpy.random.normal(prob, 0.009))))


buckets = [list(range(SIZE+1)), [0]*(SIZE + 1), [0]*(SIZE+1)]
for test in range(TESTS):
    print(test)
    buckets[1][discrete_tests(SIZE, PROBABILITY)] += 1
    buckets[2][normal(SIZE, PROBABILITY)] += 1
middle = round(PROBABILITY * SIZE)
distances = [int(100 * float(sum(buckets[1][middle - i: middle + i+1])) / float(TESTS)) for i in range(min(middle, SIZE - middle - 1))]
for i in range(len(distances)):
    print("%f: %d"%(i/SIZE, distances[i]))
buckets[2][normal(SIZE, PROBABILITY)] += 1

for j in range(3):
    for i in range(SIZE):
        print("%8d|"%(buckets[j][i]), end="")
    print()