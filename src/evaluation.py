import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
from PIL import Image

def evaluate(model, test_loader, criterion, save_dir):
    """
    Evaluates the model on the test dataset.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function used for evaluation.
        save_dir (str): Directory path to save evaluation results.

    Returns:
        tuple: A tuple containing:
            - all_labels (list): List of true labels.
            - all_predictions (list): List of predicted labels.
            - all_probabilities (list): List of probability distributions for each prediction.
            - test_loss (float): The average test loss.
            - test_acc (float): The test accuracy percentage.
    """
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    test_images = []
    
    with torch.no_grad():
        correct = 0
        total = 0
        test_loss = 0.0
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            test_images.extend(images.cpu().numpy())  
        
        test_loss /= len(test_loader)
        test_acc = 100 * correct / total
        
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
        
        save_test_results(
            all_labels, 
            all_predictions, 
            all_probabilities, 
            test_images, 
            test_loss, 
            test_acc, 
            save_dir
        )
        
        return all_labels, all_predictions, all_probabilities, test_loss, test_acc

def save_test_results(true_labels, pred_labels, probabilities, test_images, test_loss, test_acc, save_dir):
    """
    Saves the test results including metrics, predictions, images, and confusion matrix.

    Args:
        true_labels (list): List of true labels.
        pred_labels (list): List of predicted labels.
        probabilities (list): List of probability distributions for each prediction.
        test_images (list): List of test images.
        test_loss (float): The average test loss.
        test_acc (float): The test accuracy percentage.
        save_dir (str): Directory path to save results.
    """
    # Save metrics
    test_metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'correct_predictions': correct,
        'total_samples': total
    }
    with open(f'{save_dir}/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    # Save predictions
    results_df = pd.DataFrame({
        'true_label': true_labels,
        'predicted_label': pred_labels,
        'probabilities': list(probabilities)
    })
    results_df.to_csv(f'{save_dir}/test_predictions.csv', index=False)
    
    # Save sample test images
    os.makedirs(f'{save_dir}/test_images', exist_ok=True)
    for i in range(min(20, len(test_images))):
        img = test_images[i].squeeze()
        img = (img * 255).astype(np.uint8)  
        im = Image.fromarray(img)
        
        true_label = true_labels[i]
        pred_label = pred_labels[i]
        prob = max(probabilities[i]) * 100
        
        im.save(f'{save_dir}/test_images/test_{i}_true_{true_label}_pred_{pred_label}_prob_{prob:.1f}.png')
    
    # Classification report
    class_report = classification_report(true_labels, pred_labels, target_names=[str(i) for i in range(10)], output_dict=True)
    with open(f'{save_dir}/classification_report.json', 'w') as f:
        json.dump(class_report, f, indent=4)
    
    # Confusion matrix
    conf_mat = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[str(i) for i in range(10)], 
                yticklabels=[str(i) for i in range(10)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.close()
    
    pd.DataFrame(conf_mat).to_csv(f'{save_dir}/confusion_matrix.csv')

def save_test_predictions(test_loader, model, save_dir, num_images=16):
    """
    Saves a visualization of test predictions by displaying sample images along with predicted labels.

    Args:
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        model (torch.nn.Module): The trained model.
        save_dir (str): Directory path to save the visualization.
        num_images (int, optional): Number of test images to visualize. Default is 16.
    """
    model.eval()
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()
    probabilities = probabilities.cpu()
    
    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        prob_percent = probabilities[i][predicted[i]] * 100
        plt.title(f'True: {labels[i]}\nPred: {predicted[i]} ({prob_percent:.1f}%)')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/test_predictions_visualization.png')
    plt.close()
