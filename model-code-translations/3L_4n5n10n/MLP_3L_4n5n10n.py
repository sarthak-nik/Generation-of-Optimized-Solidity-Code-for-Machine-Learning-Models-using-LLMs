import torch
from torch import nn
import torch.nn.functional as F

class MLP_3L_4n5n10n(nn.Module):
    def __init__(self, input_dim):
        super(MLP_3L_4n5n10n_s, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4)  # Input to Hidden Layer with 4 neurons
        self.fc2 = nn.Linear(4, 5)         # Hidden Layer 1 to hidden Layer 2
        self.fc3 = nn.Linear(5, 10)         # Hidden Layer 2 to Output

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
