import torch
from torch import nn
import torch.nn.functional as F

class MLP_2L_4n10n(nn.Module):
    def __init__(self, input_dim):
        super(MLP_2L_4n10n_s, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4)  # Input to Hidden Layer with 4 neurons
        self.fc2 = nn.Linear(4, 10)         # Hidden Layer to Output

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
