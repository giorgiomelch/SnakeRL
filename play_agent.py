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

def dqn_agent_e(speed=10):
    file = "DQN_saved_model/easy_state/model_DDQN.keras"
    env_visual = enviroment.LinearStateSnakeGame(visual=True, speed=speed)
    model = keras.models.load_model(file)
    game_over = False
    state = env_visual.get_state()
    while not game_over:
        action = np.argmax(model(state[np.newaxis])[0])
        final_move = [0,0,0]
        final_move[action] = 1
        state, _, game_over, score = env_visual.play_step(final_move)
    env_visual.close_pygame()
    print(f"Score: {score}\n", end="")

def dqn_agent_m(visual_range, speed=5):
    if visual_range==7:
        file = "./DQN_saved_model/matrix_state/bmodel_DDQN.keras"
    elif visual_range==9:
        file = "./DQN_saved_model/matrix_state/model_9_3h_DDQN.keras"
    elif visual_range==11:
        file = "./DQN_saved_model/matrix_state/model_11_3h_DDQN.keras"
    env_visual = enviroment.MatrixStateSnakeGame(visual=True, speed=speed, visual_range=visual_range)
    model = keras.models.load_model(file)
    game_over = False
    state = env_visual.get_state()
    while not game_over:
        action = np.argmax(model(state[np.newaxis])[0])
        final_move = [0,0,0,0]
        final_move[action] = 1
        state, _, game_over, score = env_visual.play_step(final_move)
    env_visual.close_pygame()
    print(f"Score: {score}\n", end="")

def display_agent_options():
    print("Seleziona un agente di reinforcement learning da eseguire:")
    print("1. Tabular QLearning, input di 11 elementi")
    print("2. DQN, input di 11 elementi")
    print("3. DQN, input matrice 7x7")
    print("4. DQN, input matrice 9x9")
    print("5. DQN, input matrice 11x11")
    print("0. Esci")

def main():
    while True:
        display_agent_options()
        choice = input("Inserisci il numero dell'agente che desideri eseguire: ")

        if choice in {"1", "2", "3", "4", "5"}:
            try:
                speed = float(input("Inserisci la velocità dello Snake (es. 10.0): "))
            except ValueError:
                print("Input non valido. Devi inserire un numero.")
                continue

        if choice == "1":
            tab_agent(speed)

        elif choice == "2":
            # Esegui l'agente DDQN
            dqn_agent_e(speed)

        elif choice == "3":
            dqn_agent_m(7, speed)

        elif choice == "4":
            dqn_agent_m(9, speed)

        elif choice == "5":
            dqn_agent_m(11, speed)

        elif choice == "0":
            print("Uscita dal programma.")
            break

        else:
            print("Opzione non valida. Riprova.")

if __name__ == "__main__":
    main()
