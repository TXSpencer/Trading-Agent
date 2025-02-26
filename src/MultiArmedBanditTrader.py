import numpy as np

class MultiArmedBanditTrader:

    def __init__(self, epsilon=0.1):
    
        self.actions = ['Buy', 'Sell', 'Hold']
        self.epsilon = epsilon
        self.q_values = {action:0.0 for action in self.actions} # Rewards estimation
        self.action_counts = {action:0 for action in self.actions} # Number of uses

    def choose_action(self):
        """
        Select an action regarding the epsilon-greedy strategy
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions) # Exploration (random choice)
        else:
            return max(self.q_values, key=self.q_values.get) # Exploitation (optimal choice)
    
    def update_q_values(self, action, reward):
        """
        Update Q(a) using an incremental average
        """
        self.action_counts[action] += 1
        alpha = 0.1 # 1 / self.action_counts[action] # adaptative learning rate
        self.q_values[action] += alpha * (reward - self.q_values[action]) # weighted moving average