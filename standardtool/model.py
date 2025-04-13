"""
Roll Number: 24BM6JP04
Project Number: DPNN (6)
Project Title: Detection of Cardiovascular Disease using Neural Network
"""
import torch.nn as nn
import torch.nn.functional as F

class ANNModel(nn.Module):
    def __init__(self, input_size, hidden_layers=2, hidden_dim=32, output_dim=2):
        super(ANNModel, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_dim))

        for _ in range(hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)
