import torch
import numpy as np
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet

def get_state(game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state_simple = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]
        
        d1, d2, d3 = SnakeGameAI.collision_distance(game)
        state_distance = [
            d1,
            d2,
            d3,
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state_simple, dtype=int)

def get_action(model, state):
    final_move = [0,0,0]

    state0 = torch.tensor(state, dtype=torch.float)
    prediction = model(state0) # predict
    move = torch.argmax(prediction).item()
    final_move[move] = 1

    return final_move

def play():
    model = Linear_QNet(11, 256, 3)
    model.load_state_dict(torch.load("model/model.pth", weights_only=True), strict=False)
    model.eval()

    game = SnakeGameAI()
    done = False
    while done == False:
        # get old state
        state_old = get_state(game)

        # get move
        final_move = get_action(model, state_old) 

        # perform move and get new state
        _, done, score = game.play_step(final_move)

        if done:
            print("Score", score)

if __name__ == "__main__":
    play()