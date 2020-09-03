import torch
import torch.nn as nn

class SVM(nn.Module):
    def __init__(self, input_dimension, bias=False):
        super().__init__()
        self.fully_connected = nn.Linear(input_dimension, 1, bias=bias)
        
    def forward(self, x):
        fwd = self.fully_connected(x)  # Forward pass
        return fwd
