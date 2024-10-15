import numpy as np

class CircularBuffer:
    def __init__(self, max_size):
        self.buffer = np.empty(max_size, dtype=object)
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.randint(self.size, size=batch_size)
        return self.buffer[indices]
    
    def sample_experiences(self, batch_size):
        batch = self.sample(batch_size)
        states, actions, rewards, next_states, game_over = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, game_over
    
class PrioritizedReplayBuffer:
    def __init__(self, max_size, zeta=0.6):
        self.max_size = max_size
        self.buffer = []
        self.zeta = zeta
        self.priorities =  np.zeros((max_size,), dtype=np.float32)
        self.index = 0

    def append(self, experience):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.index] = experience
        self.priorities[self.index] = max_priority
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.max_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.index]
        
        probabilities = priorities ** self.zeta
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        sampling_probabilities = probabilities[indices]
        weights = (total * sampling_probabilities) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        return states, actions, rewards, next_states, dones, indices, weights


    def update_priorities(self, batch_indices, batch_errors):
        for i, error in zip(batch_indices, batch_errors):
            self.priorities[i] = np.abs(error) + 1e-5