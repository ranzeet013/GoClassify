import torch.optim as optim
import time
import pandas as pd
import matplotlib.pyplot as plt

def train(model, train_loader, criterion, optimizer, num_epochs, save_dir):
    """
    Trains the given model using the provided data loader, loss function, and optimizer.
    
    Parameters:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of epochs to train.
        save_dir (str): Directory to save training history and plots.
    
    Returns:
        dict: Training history containing epoch numbers, training loss, accuracy, and time per epoch.
    """
    model.train()
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'epoch_time': []
    }
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_time = time.time() - start_time
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['epoch_time'].append(epoch_time)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {epoch_loss:.4f}, '
              f'Accuracy: {epoch_acc:.2f}%, '
              f'Time: {epoch_time:.2f}s')

    save_training_results(history, save_dir)
    return history

def save_training_results(history, save_dir):
    """
    Saves training history to a CSV file and generates training loss and accuracy plots.
    
    Parameters:
        history (dict): Training history containing loss, accuracy, and epoch times.
        save_dir (str): Directory to save the CSV file and plots.
    """
    pd.DataFrame(history).to_csv(f'{save_dir}/training_history.csv', index=False)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['train_acc'], label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(f'{save_dir}/training_curves.png')
    plt.close()
