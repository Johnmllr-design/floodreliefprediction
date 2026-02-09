import torch.nn as nn
from torch import Tensor

class flood_comp_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Linear(6, 10)
        self.second_layer = nn.Linear(10, 30)
        self.final_layer = nn.Linear(30, 1)
        self.activation_fn = nn.ReLU()

    def forward(self, input_tensor: Tensor):
        first_output = self.first_layer(input_tensor)
        second_output = self.activation_fn(self.second_layer(first_output))
        prediction = self.activation_fn(self.final_layer(second_output))
        return prediction

    