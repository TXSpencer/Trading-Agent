import torch
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128) # input dim is 5 (features from TSLA_stock)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim) # 3 possible output (Hold, Buy, Sell)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return self.fc3(x) # No activation function here because we are looking for a Q-value
    