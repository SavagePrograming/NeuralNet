import random
from typing import Type, Union, List, Tuple, Callable
import numpy
import pygame


def distance_formula(a: Type[list], b: Type[list]):
    return numpy.linalg.norm(numpy.subtract(a, b))


def sigmoid(x: Union[int, float, numpy.array]) -> Union[int, float, numpy.array]:
    return numpy.divide(1.0, numpy.add(1.0, numpy.exp(numpy.negative(x))))


def sigmoid_der(array: Union[int, float, numpy.array]) -> Union[int, float, numpy.array]:
    return numpy.multiply(numpy.subtract(1.0, array), array)


def tanh_derivative(x: Union[int, float, numpy.array]) -> Union[int, float, numpy.array]:
    return numpy.subtract(1.0, numpy.square(x))


tanh = numpy.tanh


def rand(num: Union[int, float]) -> Union[float]:
    return random.random() + num


randomize = numpy.vectorize(rand)


def color_formula(x: Union[int, float]) -> Tuple[int, int, int]:
    return 0, int(x * 255.), 0


def map_helper(formula: Callable, *args: list):
    return list(map(formula, *args))


def map_helper_clean(formula: Callable, *args: list):
    return any(map(formula, *args))


def color_formula_line(x: Union[int, float]) -> Tuple[int, int, int]:
    return 255. - 255. * sigmoid(x), 125, 255. * sigmoid(x)


def color_formula_line_helper(x: List[Union[int, float]]) -> List[Tuple[int, int, int]]:
    return list(map(color_formula_line, x))


def draw_circle(screen_range: pygame.Surface, color_range: Tuple[int, int, int],
                in_range_loc: Tuple[int, int], radius_range: int):
    pygame.draw.circle(screen_range, color_range, in_range_loc, radius_range)


def draw_circle_helper(screen_range: List[pygame.Surface], color_range: List[Tuple[int, int, int]],
                       in_range_loc: List[Tuple[int, int]], radius_range: List[int]):
    any(map(draw_circle, screen_range, color_range, in_range_loc, radius_range))


def draw_line_enable(screen: pygame.Surface, color: Tuple[int, int, int], start_pos: Tuple[int, int],
                     end_pos: Tuple[int, int], width: int, enable: bool):
    if enable:
        pygame.draw.line(screen, color, start_pos, end_pos, width)


def draw_line_helper(screen: List[pygame.Surface], color: List[Tuple[int, int, int]],
                     start_pos: List[Tuple[int, int]],
                     end_pos: List[Tuple[int, int]], width: List[int], enable: List[bool]):
    any(map(draw_line_enable, screen, color, start_pos, end_pos, width, enable))


def dim(a: list):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])


def test_one(prob):
    return 1 if random.random() < prob else 0

def discrete_tests(size, prob):
    return sum(map(test_one, [prob] * size))