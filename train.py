import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTConvNet
from flask import Flask, render_template, jsonify
import threading
import json
from tqdm import tqdm
import numpy as np

app = Flask(__name__)

# Global variables to store training progress
training_losses = []
training_accuracies = []
current_accuracy = 0
is_training_complete = False
test_results = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_progress')
def get_progress():
    return jsonify({
        'losses': training_losses,
        'accuracies': training_accuracies,
        'current_accuracy': current_accuracy,
        'completed': is_training_complete,
        'test_results': test_results
    })

def train_model():
    global training_losses, training_accuracies, current_accuracy, is_training_complete, test_results
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"{'='*50}\n")
    
    # Hyperparameters
    num_epochs = 10
    batch_size = 512  # Increased batch size for larger GPU
    learning_rate = 0.001
    
    print("Loading datasets...")
    # MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Dataset loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    
    model = MNISTConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("\nStarting training...")
    def calculate_accuracy(model, data_loader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        batch_count = 0
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            if batch_count % 10 == 0:
                avg_loss = running_loss / batch_count
                current_acc = calculate_accuracy(model, test_loader, device)
                
                training_losses.append(avg_loss)
                training_accuracies.append(current_acc)
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'accuracy': f'{current_acc:.2f}%'
                })
        
        # Calculate final accuracy for epoch
        current_accuracy = calculate_accuracy(model, test_loader, device)
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Average Loss: {running_loss/len(train_loader):.4f}')
        print(f'Accuracy: {current_accuracy:.2f}%\n')
    
    print("Training completed! Getting test results...")
    # Get results for 10 random test images
    model.eval()
    test_images = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            for i in range(len(images)):
                if len(test_images) < 10:
                    test_images.append({
                        'image': images[i].cpu().numpy().tolist(),
                        'true_label': int(labels[i]),
                        'predicted_label': int(predicted[i])
                    })
            if len(test_images) >= 10:
                break
    
    test_results = test_images
    is_training_complete = True
    print("Test results ready! Check the web interface for visualizations.")

if __name__ == '__main__':
    print("\nStarting MNIST CNN Training Server")
    print("Please open http://localhost:5000 in your web browser to view training progress")
    
    # Start training in a separate thread
    training_thread = threading.Thread(target=train_model)
    training_thread.start()
    
    # Start Flask server
    app.run(debug=False) 