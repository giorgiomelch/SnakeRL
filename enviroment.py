import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import time

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')
BLOCK_SIZE = 20

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN1 = (0, 255, 100)
GREEN2 = (0, 255, 200)

class LinearStateSnakeGame:

    def __init__(self, w=400, h=400, visual=True, speed=0):
        self.w = w
        self.h = h
        self.state_shape = [11]
        self.n_actions = 3
        self.visual = visual
        self.speed = speed
        self.reset()

        if self.visual:
            # Inizializzazione Pygame per la modalità visiva
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 25)
            self.update_ui()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        if self.visual:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        # 1. move
        self.move(action)
        self.snake.insert(0, self.head) # update the head
        # 2. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 50*len(self.snake):
            game_over = True
            reward = -10
            return self.get_state(), reward, game_over, self.score
        # 3. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        if self.visual:
            self.update_ui()
            if self.speed != 0:
                self.clock.tick(self.speed)
        # 5. return game over and score
        new_state = self.get_state()
        return new_state, reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0: 
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def get_state(self):
        head = self.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or 
            (dir_l and self.is_collision(point_l)) or 
            (dir_u and self.is_collision(point_u)) or 
            (dir_d and self.is_collision(point_d)),
            # Danger right
            (dir_u and self.is_collision(point_r)) or 
            (dir_d and self.is_collision(point_l)) or 
            (dir_l and self.is_collision(point_u)) or 
            (dir_r and self.is_collision(point_d)),
            # Danger left
            (dir_d and self.is_collision(point_r)) or 
            (dir_u and self.is_collision(point_l)) or 
            (dir_r and self.is_collision(point_u)) or 
            (dir_l and self.is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location 
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y  # food down
            ]
        return np.array(state, dtype=int)
    
    def update_ui(self):
        self.display.fill(BLACK)
        for i, pt in enumerate(self.snake):
            if i == 0:
                pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            else:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def close_pygame(self):
        if self.visual:
            time.sleep(2)
            pygame.quit()

    def set_fruit(self, x, y):
        self.food = Point(x, y)

    def set_test_uno(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-(1*BLOCK_SIZE), self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y),
                      Point(self.head.x-(3*BLOCK_SIZE), self.head.y),
                      Point(self.head.x-(3*BLOCK_SIZE), self.head.y-(1*BLOCK_SIZE)),
                      Point(self.head.x-(3*BLOCK_SIZE), self.head.y-(2*BLOCK_SIZE)),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y-(2*BLOCK_SIZE)),
                      Point(self.head.x-(1*BLOCK_SIZE), self.head.y-(2*BLOCK_SIZE)),
                      Point(self.head.x, self.head.y-(2*BLOCK_SIZE)),
                      Point(self.head.x+(1*BLOCK_SIZE), self.head.y-(2*BLOCK_SIZE)),
                      Point(self.head.x+(2*BLOCK_SIZE), self.head.y-(2*BLOCK_SIZE)),
                      Point(self.head.x+(3*BLOCK_SIZE), self.head.y-(2*BLOCK_SIZE)),
                      Point(self.head.x+(4*BLOCK_SIZE), self.head.y-(2*BLOCK_SIZE)),
                      Point(self.head.x+(5*BLOCK_SIZE), self.head.y-(2*BLOCK_SIZE)),
                      Point(self.head.x+(6*BLOCK_SIZE), self.head.y-(2*BLOCK_SIZE)),
                      Point(self.head.x+(7*BLOCK_SIZE), self.head.y-(2*BLOCK_SIZE)),
                      Point(self.head.x+(8*BLOCK_SIZE), self.head.y-(2*BLOCK_SIZE)),
                      Point(self.head.x+(9*BLOCK_SIZE), self.head.y-(2*BLOCK_SIZE))]
        self.set_fruit(self.head.x-(2*BLOCK_SIZE), self.head.y-(1*BLOCK_SIZE))

    def set_test_due(self):
        self.direction = Direction.LEFT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x+(1*BLOCK_SIZE), self.head.y),
                      Point(self.head.x+(2*BLOCK_SIZE), self.head.y),
                      Point(self.head.x+(3*BLOCK_SIZE), self.head.y),
                      Point(self.head.x+(3*BLOCK_SIZE), self.head.y-(1*BLOCK_SIZE)),
                      Point(self.head.x+(2*BLOCK_SIZE), self.head.y-(1*BLOCK_SIZE)),
                      Point(self.head.x+(1*BLOCK_SIZE), self.head.y-(1*BLOCK_SIZE)),
                      Point(self.head.x, self.head.y-(1*BLOCK_SIZE)),
                      Point(self.head.x-(1*BLOCK_SIZE), self.head.y-(1*BLOCK_SIZE)),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y-(1*BLOCK_SIZE)),
                      Point(self.head.x-(3*BLOCK_SIZE), self.head.y-(1*BLOCK_SIZE)),
                      Point(self.head.x-(4*BLOCK_SIZE), self.head.y-(1*BLOCK_SIZE)),
                      Point(self.head.x-(4*BLOCK_SIZE), self.head.y),
                      Point(self.head.x-(4*BLOCK_SIZE), self.head.y+(1*BLOCK_SIZE)),
                      Point(self.head.x-(4*BLOCK_SIZE), self.head.y+(2*BLOCK_SIZE)),
                      Point(self.head.x-(4*BLOCK_SIZE), self.head.y+(3*BLOCK_SIZE)),
                      Point(self.head.x-(4*BLOCK_SIZE), self.head.y+(3*BLOCK_SIZE)),
                      Point(self.head.x-(3*BLOCK_SIZE), self.head.y+(3*BLOCK_SIZE)),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y+(3*BLOCK_SIZE)),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y+(2*BLOCK_SIZE)),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y+(1*BLOCK_SIZE)),
                      Point(self.head.x-(1*BLOCK_SIZE), self.head.y+(1*BLOCK_SIZE)),
                      Point(self.head.x-(1*BLOCK_SIZE), self.head.y+(2*BLOCK_SIZE)),
                      Point(self.head.x-(1*BLOCK_SIZE), self.head.y+(3*BLOCK_SIZE)),
                      Point(self.head.x-(1*BLOCK_SIZE), self.head.y+(4*BLOCK_SIZE)),
                      Point(self.head.x-(1*BLOCK_SIZE), self.head.y+(5*BLOCK_SIZE))]
        self.set_fruit(self.head.x-(3*BLOCK_SIZE), self.head.y+(2*BLOCK_SIZE))

class MatrixStateSnakeGame(LinearStateSnakeGame):
    def __init__(self, w=400, h=400, visual=True, speed=0, visual_range=7):
        super().__init__(w, h, visual, speed)
        if visual_range % 2 == 0:
            raise ValueError("La dimensione della matrice deve essere dispari.")
        self.visual_range = visual_range
        self.n_actions = 4
        self.state_shape = [visual_range**2 + self.n_actions]

    def move(self, action):
        direction_array = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = np.argmax(action)
        new_dir = direction_array[idx]
        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def get_state(self):
        state = np.zeros((self.visual_range, self.visual_range), dtype=int)
        center = self.visual_range // 2
        state[center, center] = 2
        for segment in self.snake[1:]:
            x_diff = int((segment.x - self.head.x) / BLOCK_SIZE)
            y_diff = int((segment.y - self.head.y) / BLOCK_SIZE)
            if abs(x_diff) <= center and abs(y_diff) <= center:
                state[center + y_diff, center + x_diff] = 1
        for i in range(-center, center + 1):
            for j in range(-center, center + 1):
                if (self.head.x + i * BLOCK_SIZE < 0 or
                    self.head.x + i * BLOCK_SIZE >= self.w or
                    self.head.y + j * BLOCK_SIZE < 0 or
                    self.head.y + j * BLOCK_SIZE >= self.h):
                    state[center + j, center + i] = 1
        x_diff = int((self.food.x - self.head.x) / BLOCK_SIZE)
        y_diff = int((self.food.y - self.head.y) / BLOCK_SIZE)
        if abs(x_diff) <= center and abs(y_diff) <= center:
            state[center + y_diff, center + x_diff] = -1
        food_direction = [
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y   # food down
        ]
        final_state = np.concatenate((state.flatten(), food_direction))
        return final_state