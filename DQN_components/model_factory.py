import tensorflow as tf
import keras
import os


def Linear_QNet(input_shape, n_output, units):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    for unit in units:
        model.add(keras.layers.Dense(unit, activation='relu'))
    model.add(keras.layers.Dense(n_output))
    return model

class QNetwork:
    def __init__(self, lr, gamma, input_shape, n_output, units):
        self.model = Linear_QNet(input_shape, n_output, units)
        self.gamma = gamma
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = keras.losses.mean_squared_error
        self.n_output = n_output
        
    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        next_Q_values = self.model(next_states)
        max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
        # Equazione di Bellman: Q value = reward + discount factor * expected future reward
        target_Q_values = rewards + (1 - dones) * self.gamma * max_next_Q_values
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)  
            Q_values = tf.reduce_sum(all_Q_values * actions, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def save_model(self, directory_path, file_name):
        if not os.path.exists(directory_path):
            print(f"Errore: la directory '{directory_path}' non esiste. Modello non salvato.")
            return
        full_path = os.path.join(directory_path, file_name)
        try:
            self.model.save(full_path)
        except Exception as e:
            print(f"Errore durante il salvataggio del modello: {e}")

class DoubleQNetwork(QNetwork):
    def __init__(self, lr, gamma, input_shape, n_output, units):
        super().__init__(lr, gamma, input_shape, n_output, units)
        self.online_model = self.model
        self.target_model = keras.models.clone_model(self.online_model)
        self.target_model.set_weights(self.online_model.get_weights())

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        next_Q_values = self.online_model(next_states)
        # Double DQN: l'online model sceglie l'azione dei prossimi stati ma i Q-Value sono stimati da target_model
        best_next_actions = tf.argmax(next_Q_values, axis=1)
        mask_for_target = tf.one_hot(best_next_actions, self.n_output)
        max_next_Q_values = tf.reduce_sum(self.target_model(next_states) * mask_for_target, axis=1)
        target_Q_values = rewards + (1 - dones) * self.gamma * max_next_Q_values
        with tf.GradientTape() as tape:
            all_Q_values = self.online_model(states)  
            Q_values = tf.reduce_sum(all_Q_values * actions, axis=1, keepdims=False)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.online_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_model.trainable_variables))
    
    def update_weights(self):
        self.target_model.set_weights(self.online_model.get_weights())


class PER_QNetwork(QNetwork):
    def __init__(self, lr, gamma, input_shape, n_output, units):
        super().__init__(lr, gamma, input_shape, n_output, units)

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones, weights):
        next_Q_values = self.model(next_states)
        max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
        target_Q_values = rewards + (1 - dones) * self.gamma * max_next_Q_values
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)  
            Q_values = tf.reduce_sum(all_Q_values * actions, axis=1, keepdims=False)
            loss = tf.reduce_mean(weights * self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        td_errors = tf.abs(tf.subtract(target_Q_values, Q_values))
        return td_errors

class DDQN_PER_QNetwork(DoubleQNetwork):
    def __init__(self, lr, gamma, input_shape, n_output, units):
        super().__init__(lr, gamma, input_shape, n_output, units)

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones, weights):
        next_Q_values = self.online_model(next_states)
        # Double DQN: l'online model sceglie l'azione dei prossimi stati ma i Q-Value sono stimati da target_model
        best_next_actions = tf.argmax(next_Q_values, axis=1)
        mask_for_target = tf.one_hot(best_next_actions, self.n_output)
        max_next_Q_values = tf.reduce_sum(self.target_model(next_states) * mask_for_target, axis=1)
        # Equazione di Bellman: Q value = reward + discount factor * expected future reward
        target_Q_values = rewards + (1 - dones) * self.gamma * max_next_Q_values
        with tf.GradientTape() as tape:
            all_Q_values = self.online_model(states)  
            Q_values = tf.reduce_sum(all_Q_values * actions, axis=1, keepdims=False)
            loss = tf.reduce_mean(weights * self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.online_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_model.trainable_variables))
        td_errors = tf.abs(tf.subtract(target_Q_values, Q_values))
        return td_errors