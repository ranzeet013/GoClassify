import torch.nn as nn

class MNISTClassifier(nn.Module):
    """
    A simple feedforward neural network for MNIST digit classification.

    The network consists of:
    - A flattening layer to convert images into a 1D vector
    - Two fully connected layers with ReLU activation
    - A final fully connected layer for classification into 10 digits (0-9)
    """
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10)
        """
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def initialize_model(device):
    """
    Initializes the MNIST classification model and moves it to the specified device.

    Args:
        device (torch.device): The device to which the model should be moved (CPU/GPU)
    
    Returns:
        MNISTClassifier: The initialized model on the specified device.
    """
    model = MNISTClassifier().to(device)
    return model
