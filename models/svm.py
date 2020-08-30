import torch
import torch.nn as nn

class SVM(nn.Module):
    """
    Linear Support Vector Machine
    -----------------------------
    This SVM is a subclass of the PyTorch nn module that
    implements the Linear  function. The  size  of  each 
    input sample is input_dimension and output sample  is 1.
    """
    def __init__(self, input_dimension):
        super().__init__()  # Call the init function of nn.Module
        self.fully_connected = nn.Linear(input_dimension, 1, bias=False)  # Implement the Linear function
        
    def forward(self, x):
        fwd = self.fully_connected(x)  # Forward pass
        return fwd
