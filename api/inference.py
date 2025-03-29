import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json

class MNISTClassifier(nn.Module):
    """
    A simple feedforward neural network for MNIST digit classification.
    The model consists of:
    - A flattening layer
    - Fully connected layers: (28x28 -> 512 -> 256 -> 10)
    - ReLU activations between layers
    """
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10).
        """
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def load_model():
    """
    Loads the trained MNIST model from the specified file path.

    Returns:
        MNISTClassifier: The trained model in evaluation mode.
    """
    model_path = "/Users/Raneet/Desktop/image-classifications/outputs/models/classification_model_weights.pth"
    model = MNISTClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def preprocess_image(image_path):
    """
    Preprocesses an input image for MNIST classification.
    
    Steps:
    - Convert to grayscale
    - Resize to 28x28
    - Convert to tensor
    - Normalize using MNIST mean and std

    Args:
        image_path (str): Path to the input image.
    
    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 1, 28, 28).
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict_image(model, image_tensor):
    """
    Predicts the digit class for a given preprocessed image tensor.
    
    Args:
        model (nn.Module): Trained MNIST classification model.
        image_tensor (torch.Tensor): Preprocessed image tensor.
    
    Returns:
        tuple: Predicted digit (int) and class probabilities (dict).
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return predicted.item(), probabilities.squeeze().numpy()

if __name__ == "__main__":
    image_path = sys.argv[1]
    model = load_model()
    image_tensor = preprocess_image(image_path)
    prediction, probabilities = predict_image(model, image_tensor)

    # Convert to JSON format
    result = {
        "predicted_digit": prediction,
        "probabilities": {str(i): float(probabilities[i]) for i in range(10)}
    }

    print(json.dumps(result))
