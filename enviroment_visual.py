import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font(None, 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
GREEN1 = (0, 255, 100)
GREEN2 = (0, 255, 200)

BLOCK_SIZE = 20

class SnakeGameAI:

    def __init__(self, w=400, h=400, speed=8):
        self.w = w
        self.h = h
        self.speed = speed
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
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
        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # 1. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
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
        # 4. update ui and clock
        self._update_ui()
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

    def _update_ui(self):
        self.display.fill(BLACK)

        for i, pt in enumerate(self.snake):
            if i == 0:
                pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            else:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
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
    
    def get_matrix_state(self):
        state = np.zeros((self.h // BLOCK_SIZE, self.w // BLOCK_SIZE))
        for point in self.snake[1:]:
            x_index = int(point.x // BLOCK_SIZE)
            y_index = int(point.y // BLOCK_SIZE)
            state[y_index][x_index] = 1  # Corpo rappresentato da 1
        head = self.snake[0]
        x_index = int(head.x // BLOCK_SIZE)
        y_index = int(head.y // BLOCK_SIZE)
        if x_index < self.w/BLOCK_SIZE and y_index < self.h/BLOCK_SIZE:
            state[y_index][x_index] = 2  # Testa rappresentata da 2
        x_index = int(self.food.x // BLOCK_SIZE)
        y_index = int(self.food.y // BLOCK_SIZE)
        state[y_index][x_index] = 3  # Cibo rappresentato da -1
        return state
    
    def play_step_m(self, action):
        self.frame_iteration += 1
        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # 1. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        # 2. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 50*len(self.snake):
            game_over = True
            reward = -10
            return self.get_matrix_state(), reward, game_over, self.score
        # 3. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        # 4. update ui and clock
        self._update_ui()
        if self.speed != 0:
            self.clock.tick(self.speed)
        # 5. return game over and score
        new_state = self.get_matrix_state()
        return new_state, reward, game_over, self.score

    def quit(self):      
        pygame.quit()