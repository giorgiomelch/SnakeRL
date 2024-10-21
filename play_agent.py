import argparse
import enviroment
import numpy as np
import keras

def state_to_index(state_vector):
    pericoli = state_vector[0] * 4 + state_vector[1] * 2 + state_vector[2]
    direzione_corrente = state_vector[3] * 3 + state_vector[4] * 2 + state_vector[5] * 1
    posizione_frutto = np.mod(state_vector[7] * 6 + state_vector[8] * 3 + state_vector[9] * 2 + state_vector[10], 8)
    return pericoli * 32 + direzione_corrente * 8 + posizione_frutto
def initialize_QTable():
    return np.zeros((256, 3))


def tab_agent(speed=10):
    file = 'Q_table/ExplorationFunction/1000000step.npy' 
    Q_table = np.load(file)
    env_visual = enviroment.LinearStateSnakeGame(visual=True, speed=speed)
    env_visual.reset()
    game_over = False
    while not game_over:
        state= env_visual.get_state()
        row_Q_value = state_to_index(state)
        action = np.argmax(Q_table[row_Q_value])
        final_move = [0,0,0]
        final_move[action] = 1
        _, _, game_over, score = env_visual.play_step(final_move)
        
    env_visual.close_pygame()
    print(f"Score: {score}\n", end="")

def lnn_agent(speed=10):
    file = "DQN_saved_model/LNN/model.keras"
    env_visual = enviroment.LinearStateSnakeGame(visual=True, speed=speed)
    model = keras.models.load_model(file)
    game_over = False
    state = env_visual.get_state()
    while not game_over:
        action = np.argmax(model(state[np.newaxis])[0])
        final_move = [0,0,0]
        final_move[action] = 1
        state, _, game_over, score = env_visual.play_step(final_move)
    env_visual.quit()
    print(f"Score: {score}\n", end="")

def lnn_agent_m(speed=5):
    file = "./DQN_saved_model/matrix_state/model_DDQN.keras"
    env_visual = enviroment.MatrixStateSnakeGame(visual=True, speed=speed)
    model = keras.models.load_model(file)
    game_over = False
    state = env_visual.get_state()
    while not game_over:
        action = np.argmax(model(state[np.newaxis])[0])
        final_move = [0,0,0,0]
        final_move[action] = 1
        state, _, game_over, score = env_visual.play_step(final_move)
        matrix = state[:49]
        matrix[matrix == -1] = 5
        matrix = matrix.reshape(7, 7)
        for i in matrix:
            print(i)
        print("\n\n\n")
    env_visual.close_pygame()
    print(f"Score: {score}\n", end="")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose which Snake agent to run.")
    parser.add_argument('agent', choices=['tab', 'lnn'], help="Choose the agent: 'tab' or 'lnn'")
    parser.add_argument('--speed', type=int, help="Speed of the Snake game", default=10)

    args = parser.parse_args()
    
    if args.agent == 'tab':
        tab_agent(speed=args.speed)
    elif args.agent == 'lnn':
        lnn_agent_m(speed=args.speed)