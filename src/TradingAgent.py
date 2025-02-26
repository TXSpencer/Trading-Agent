from collections import deque
from DQN import DQN
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random


class TradingAgent:
    
    def __init__(self, state_size, action_size, feature_dim, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99, 
                 lr=0.001, batch_size=32):
        
        self.state_size = state_size
        self.action_size = action_size
        self.feature_dim = feature_dim
        self.memory = deque(maxlen=2000) # Replay buffer
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.model = DQN(self.feature_dim, action_size)
        self.target_model = DQN(self.feature_dim, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()


    def choose_action(self, state):
        
        if np.random.rand() <= self.epsilon:
            return np.random.choice([0, 1, 2])
        state = torch.FloatTensor(state).unsqueeze(0)
        #print(state.shape)
        q_values = self.model(state)
        
        if q_values.shape[1] > 0:
            q_values = q_values[:, -1, :] # On prend le dernier timestamp
        else:
            q_values = torch.tensor([1, 0, 0]) # if q_values tensor empty, assign hold by default

        return torch.argmax(q_values).item()


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    

    def replay(self):
        """
        Uses the experience stored in the agent memory to update the model
        """

        if len(self.memory) < self.batch_size:
            return

        mini_batch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in mini_batch:

            target = reward
            
            if not done:
                q_next = self.target_model(torch.FloatTensor(next_state).unsqueeze(0))
                q_next = q_next[:, -1, :]

                target += self.gamma * torch.max(q_next).item() # Equation de Bellman

            q_values = self.model(torch.FloatTensor(state))

            # Update chosen action's q_value
            q_values[0, action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(q_values, self.model(torch.FloatTensor(state)))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
