import torch
import json
import os
from src import (
    data_loading, 
    model, 
    training, 
    evaluation
)

# Configuration
config = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'model_architecture': 'FC-512-256-10',
    'optimizer': 'Adam',
    'loss_function': 'CrossEntropyLoss'
}

# Setup directories
os.makedirs('configs', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Save config
with open('configs/config.json', 'w') as f:
    json.dump(config, f, indent=4)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('outputs/device_info.txt', 'w') as f:
    f.write(str(device))

# Data loading
def load_data(config):
    """
    Loads the MNIST dataset with transformations and returns data loaders.
    
    Args:
        config (dict): Configuration dictionary containing batch size.
    
    Returns:
        tuple: Training and testing data loaders.
    """
    train_loader, test_loader = data_loading.load_data(config)
    return train_loader, test_loader

def save_sample_images(loader, save_dir):
    """
    Saves sample images from the dataset to visualize training samples.
    
    Args:
        loader (DataLoader): DataLoader containing the dataset.
        save_dir (str): Directory to save the sample images.
    """
    data_loading.save_sample_images(loader, save_dir)

train_loader, test_loader = load_data(config)
save_sample_images(train_loader, 'outputs')

# Model setup
model = model.initialize_model(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# Save model architecture
with open('outputs/model_architecture.txt', 'w') as f:
    f.write(str(model))

# Training
print("Starting training...")
train_history = training.train(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
    config['num_epochs'], 
    'outputs'
)

# Evaluation
print("Evaluating on test set...")
true_labels, pred_labels, probabilities, test_loss, test_acc = evaluation.evaluate(
    model, 
    test_loader, 
    criterion, 
    'outputs'
)

# Save predictions visualization
evaluation.save_test_predictions(test_loader, model, 'outputs')

# Save models
torch.save(model.state_dict(), 'models/classification_model_weights.pth')
torch.save(model, 'models/classification_model_full.pth')
