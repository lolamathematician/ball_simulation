from __future__ import annotations
import arcade
import random
from GLOBVARS import *
from copy import copy
import math
from typing import List, NewType
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

np.seterr(all='ignore')

save_number = len(os.listdir())
HEAT_MAP_FILE = 'heat_map' + str(save_number) + '.pickle'
HEAT_MAP_RADIUS = 5
HEAT_MAP_RANGE = range(-HEAT_MAP_RADIUS, HEAT_MAP_RADIUS, 1)
HEAT_MAP = np.zeros([WINDOW_HEIGHT, WINDOW_WIDTH])
HEAT_MAP = np.array(HEAT_MAP, dtype='int')


def save_heat_map():
    file = open(HEAT_MAP_FILE, 'wb')
    pickle.dump(HEAT_MAP, file)
    file.close()


def make_heat_map_plot():
    plt.imshow(HEAT_MAP, cmap='hot', interpolation='nearest')
    plt.show()


# returns True if (x, y) is not in the window
def check_not_in_window(x, y):
    return bool((x < 0 or x >= WINDOW_WIDTH) or (y < 0 or y >= WINDOW_HEIGHT))


class Ball:

    def __init__(self, initial_x, initial_y, initial_dx, initial_dy, colour, radius):
        self.x = initial_x
        self.y = initial_y
        self.dx = initial_dx
        self.dy = initial_dy
        self.colour = colour
        self.radius = radius
        self.mass = radius
        self.right = initial_x + radius
        self.left = initial_x - radius
        self.top = initial_y + radius
        self.bottom = initial_y - radius
        self.too_right = False
        self.too_left = False
        self.too_high = False
        self.too_low = False
        # all the balls that this ball is currently overlapping with
        self.collided_balls = []
        # all the generated balls but this one
        self.other_balls = None

    def update_position(self):

        # changes direction of travel if the ball hits an edge of the window
        if not self.too_left:
            if self.left <= 0:
                self.too_left = True
                self.dx = -self.dx
        else:
            if self.left >= 0:
                self.too_left = False

        if not self.too_right:
            if self.right >= WINDOW_WIDTH:
                self.too_right = True
                self.dx = -self.dx
        else:
            if self.right <= WINDOW_WIDTH:
                self.too_right = False

        if not self.too_low:
            if self.bottom <= 0:
                self.too_low = True
                self.dy = -self.dy
        else:
            if self.bottom >= 0:
                self.too_low = False

        if not self.too_high:
            if self.top >= WINDOW_HEIGHT:
                self.too_high = True
                self.dy = -self.dy
        else:
            if self.top <= WINDOW_HEIGHT:
                self.too_high = False

        # update positions of important ball features
        self.x = self.x + self.dx
        self.y = self.y + self.dy
        self.right = self.x + self.radius
        self.left = self.x - self.radius
        self.top = self.y + self.radius
        self.bottom = self.y - self.radius

        # self.update_heat_map()

    def update_collided_status(self):
        for other_ball in self.other_balls:
            center_distance = math.sqrt((self.x - other_ball.x) ** 2 + (self.y - other_ball.y) ** 2)
            edge_distance = self.radius + other_ball.radius
            currently_collided = bool(center_distance < edge_distance)
            # adds balls that this ball is currently overlapping to both this ball and the other balls collided_balls
            # list
            if currently_collided:
                if other_ball not in self.collided_balls:
                    collision(self, other_ball)
                    self.collided_balls.append(other_ball)
                    other_ball.collided_balls.append(self)
            # checks all balls in this balls collided_balls are still overlapping with this ball and removes them if not
            else:
                if other_ball in self.collided_balls:
                    self.collided_balls.remove(other_ball)

    def get_other_balls(self, all_balls: List[Ball]):
        all_balls = copy(all_balls)
        all_balls.remove(self)
        self.other_balls = all_balls

    def update_heat_map(self):
        _range = range(-self.radius, self.radius, 1)
        for x in _range:
            for y in _range:
                distance_from_centre = math.sqrt((x - self.x)**2 + (y - self.y)**2)
                if distance_from_centre <= self.radius:
                    # _x and _y are the true (not relative) coordinates to change on the heat map
                    _x = int(self.x + x)
                    _y = int(self.y + y)
                    if not check_not_in_window(_x, _y):
                        HEAT_MAP[_y][_x] += 1


NewType('Ball', Ball)

BallList = List[Ball]

COLLISION_PAIRS = []


def make_balls():
    balls = []

    '''
    # Generate one ball with predetermined properties
    ball = Ball(285, 285, 1, 1, BALL_COLOURS[0], 10)
    balls.append(ball)
    ball = Ball(315, 315, -1, -1, BALL_COLOURS[1], 10)
    balls.append(ball)
    # '''

    # '''
    for i in range(20):
        # Generates one ball with random properties and adds it to the list of balls
        radius = random.randint(10, 30)
        initial_x_direction = random.choice([-1, 1])
        initial_y_direction = random.choice([-1, 1])
        ball = Ball(random.randint(radius, WINDOW_WIDTH - radius),
                    random.randint(radius, WINDOW_HEIGHT - radius),
                    initial_x_direction * random.random() * 5,
                    initial_y_direction * random.random() * 5,
                    random.choice(BALL_COLOURS),
                    radius)
        balls.append(ball)
    # '''

    for _ball in balls:
        _ball.get_other_balls(balls)

    return balls


BALLS = make_balls()


def check_collisions(balls: BallList):
    for ball in balls:
        ball.update_collided_status()


# Just does the mathematics of two massive bodies colliding in a totally elastic collision in two dimensions
def two_d_elastic_collision(x_1: np.array, x_2: np.array, v_1: np.array, v_2: np.array, mass_1: int, mass_2: int):
    mass_ratio = 2 * mass_2 / (mass_1 + mass_2)
    vector_numerator = np.dot(v_1 - v_2, x_1 - x_2)
    vector_denominator = sum(x * x for x in x_1 - x_2)
    vector_ratio = vector_numerator / vector_denominator
    vector_distance = x_1 - x_2
    return v_1 - mass_ratio * vector_ratio * vector_distance


def collision(ball_1: Ball, ball_2: Ball):
    x_1 = np.array([ball_1.x, ball_1.y])
    x_2 = np.array([ball_2.x, ball_2.y])
    v_1 = np.array([ball_1.dx, ball_1.dy])
    v_2 = np.array([ball_2.dx, ball_2.dy])
    temp_1 = two_d_elastic_collision(x_1, x_2, v_1, v_2, ball_1.mass, ball_2.mass)
    ball_2.dx, ball_2.dy = two_d_elastic_collision(x_2, x_1, v_2, v_1, ball_2.mass, ball_1.mass)
    ball_1.dx, ball_1.dy = temp_1


# noinspection PyUnusedLocal
def on_draw(delta_time):
    """ Draw everything """
    arcade.start_render()

    check_collisions(BALLS)

    for ball in BALLS:
        arcade.draw_circle_filled(ball.x, ball.y, ball.radius, ball.colour)
        ball.update_position()


def main():
    arcade.open_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Bouncing Balls")
    arcade.set_background_color(arcade.color.BLACK)

    # Call on_draw every 60th of a second
    arcade.schedule(on_draw, 1/60)
    arcade.run()
    save_heat_map()
    make_heat_map_plot()
