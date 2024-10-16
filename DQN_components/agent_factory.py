from . import replay_buffer 
from . import model_factory
import enviroment
import tensorflow as tf
import numpy as np

def convert_to_tensorflow(states, actions, rewards, next_states, dones):
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)
    return states, actions, rewards, next_states, dones

        
class Agent:
    def __init__(self, lr, gamma, max_memory, batch_size, input_shape, n_actions, model_units=[256], 
                 eps_greedy=True, file_name_model="model.keras", visual=False):
        self.n_games = 0
        self.epsilon = 1
        self.batch_size = batch_size
        self.memory = replay_buffer.CircularBuffer(max_size=max_memory)
        self.dqnetwork = model_factory.QNetwork(lr=lr, gamma=gamma, input_shape=input_shape, n_output=n_actions, units=model_units)
        self.env = enviroment.SnakeGameVisual_v1(speed=0) if visual else enviroment.SnakeGame_v1()
        self.exploration_policy = self.epsilon_greedy_policy if eps_greedy else self.softmax_policy
        self.file_name_model = file_name_model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(convert_to_tensorflow(state, action, reward, next_state, done))

    def train_memory(self):
        states, actions, rewards, next_states, dones = self.memory.sample_experiences(self.batch_size)
        self.dqnetwork.train_step(states, actions, rewards, next_states, dones)

    def softmax_policy(self, state):
        Q_values = self.dqnetwork.online_model(state[np.newaxis])
        if self.epsilon < 0.005: # per evitare buffer overflow
            return np.argmax(Q_values[0])
        else:
            max_Q = np.max(Q_values)
            exp_q = np.exp((Q_values - max_Q) / self.epsilon)
            probabilities = exp_q / np.sum(exp_q)
            action = np.random.choice(3, p=probabilities[0])
            return action
        
    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(3)
        else:
            Q_values = self.dqnetwork.model(state[np.newaxis])
            return np.argmax(Q_values[0])
        
    def get_action(self, state):
        final_move = [0, 0, 0]  
        move = self.exploration_policy(state)
        final_move[move] = 1
        return final_move
    
    def train_agent(self, N_GAME):
        score_list = []
        record = 0
        step=0
        n_eps_zero = int(N_GAME*0.7)
        while self.n_games < N_GAME:
            state_old = self.env.get_state()
            final_move = self.get_action(state_old)
            state_new, reward, done, score = self.env.play_step(final_move)
            self.remember(state_old, final_move, reward, state_new, done)
            if done:
                self.env.reset()
                self.n_games += 1
                self.train_memory()
                print(f"\rGame: {self.n_games}, Epsilon: {self.epsilon:3f}, Score: {score}, Record: {record}, Step eseguiti: {step}. ", end="")
                self.epsilon = max(((n_eps_zero - self.n_games) / n_eps_zero), 0)
                if score > record:
                    record = score
                    self.dqnetwork.save_model(file_name=self.file_name_model)
                score_list.append(score)
            step+=1

        if isinstance(self.env, enviroment.SnakeGameVisual):
            self.env.close_pygame()   
        return score_list
    

class Agent_DoubleDQN(Agent):
    def __init__(self, lr, gamma, max_memory, batch_size, input_shape, n_actions, model_units=[256], 
                 eps_greedy=True, file_name_model="model_DOUBLE.keras", visual=False):
        super().__init__(lr, gamma, max_memory, batch_size, input_shape, n_actions, model_units, 
                         eps_greedy, file_name_model, visual)
        self.dqnetwork = model_factory.DoubleQNetwork(lr=lr, gamma=gamma, units=model_units)

    def train_agent(self, N_GAME):
        score_list = []
        record = 0
        step=0
        n_eps_zero = int(N_GAME*0.7)
        while self.n_games < N_GAME:
            state_old = self.env.get_state()
            final_move = self.get_action(state_old)
            state_new, reward, done, score = self.env.play_step(final_move)
            self.remember(state_old, final_move, reward, state_new, done)
            if done:
                self.env.reset()
                self.n_games += 1
                self.train_memory()
                print(f"\rGame: {self.n_games}, Epsilon: {self.epsilon:3f}, Score: {score}, Record: {record}, Step eseguiti: {step}. ", end="")
                self.epsilon = max(((n_eps_zero - self.n_games) / n_eps_zero), 0)
                if score > record:
                    record = score
                    self.dqnetwork.save_model()
                if self.n_games % 5:
                    self.dqnetwork.update_weights()
                score_list.append(score)
            step+=1
        return score_list

class Agent_PER(Agent):
    def __init__(self, lr, gamma, max_memory, batch_size, input_shape, n_actions, model_units=[256], 
                 eps_greedy=True, file_name_model="model_PER.keras", visual=False):
        super().__init__(lr, gamma, max_memory, batch_size, input_shape, n_actions, model_units, 
                         eps_greedy, file_name_model, visual)
        self.memory = replay_buffer.PrioritizedReplayBuffer(max_size=max_memory, zeta=0.6)
        self.dqnetwork = model_factory.PER_QNetwork(lr=lr, gamma=gamma, units=model_units)

    def train_memory(self, beta=0.4):
        if len(self.memory.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size, beta)
        td_errors = self.dqnetwork.train_step(states, actions, rewards, next_states, dones, weights)
        self.memory.update_priorities(indices, td_errors.numpy())


class Agent_DDQN_PER(Agent_DoubleDQN):
    def __init__(self, lr, gamma, max_memory, batch_size, input_shape, n_actions, model_units=[256], 
                 eps_greedy=True, file_name_model="model_DDQN_PER.keras", visual=False):
        super().__init__(lr, gamma, max_memory, batch_size, input_shape, n_actions, model_units, 
                         eps_greedy, file_name_model, visual)
        self.memory = replay_buffer.PrioritizedReplayBuffer(max_size=max_memory, zeta=0.6)
        self.dqnetwork = model_factory.DDQN_PER_QNetwork(lr=lr, gamma=gamma, units=model_units)

    def train_memory(self, beta=0.4):
        if len(self.memory.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size, beta)
        td_errors = self.dqnetwork.train_step(states, actions, rewards, next_states, dones, weights)
        self.memory.update_priorities(indices, td_errors.numpy())