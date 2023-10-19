[Colab note book with in class provided code and my code at the end.](https://colab.research.google.com/drive/1gKXzHiPCOHs9aHgIIOVd4Y2mXAA1ztnO?usp=sharing) 

# Flower Classification: Model Training & Evaluation
## Overview
The provided code aims to classify flower images using a deep learning model. It utilizes the PyTorch library for neural network training and the wandb library to monitor and log the model's performance in real-time.

# Libraries & Dependencies
PyTorch: A popular deep learning library.
torch: Main PyTorch module.
nn: Neural network module.
optim: Optimization algorithms.
functional: Functional interface.
torchvision: Used for datasets and transformations.
NumPy: For numerical operations.
wandb: Weights & Biases, a tool for experiment tracking.
```python
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import wandb
```
### WandB Initialization
```python
wandb.init(project="flower_classification")
```
#### Device Setup
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
##### Data Preprocessing
```python
# Transforms
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load data
train_data = datasets.ImageFolder('flower_data/train', transform=train_transforms)
test_data = datasets.ImageFolder('flower_data/valid', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
```
###### Loading Data 
```python
train_data = datasets.ImageFolder('flower_data/train', transform=train_transforms)
test_data = datasets.ImageFolder('flower_data/valid', transform=test_transforms)
```
###### Data Loaders
```python
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
```
###### Setting up pretrained model
```python
model = models.densenet121(pretrained=True)
```
###### Freezing parameters
```python
for param in model.parameters():
    param.requires_grad = False
```
##### Classifier
```python
# Classifier
classifier = nn.Sequential(nn.Linear(1024, 256),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(256, 102),
                           nn.LogSoftmax(dim=1))

model.classifier = classifier
model.to(device)
```
###### Loss and Optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

##### Training & Evaluation Loop
```python
epochs = 3

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        logps = model(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Logging the metrics in real-time after each batch
        wandb.log({"Training Loss per Batch": loss.item()})

    # Evaluate on validation set after each epoch
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            test_loss += criterion(logps, labels).item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    # Logging the metrics after each epoch
    wandb.log({"Training Loss per Epoch": running_loss/len(trainloader),
               "Validation Loss per Epoch": test_loss/len(testloader),
               "Validation Accuracy per Epoch": accuracy/len(testloader)})

    print(f"Epoch {epoch+1}/{epochs}.. "
          f"Train loss: {running_loss/len(trainloader):.3f}.. "
          f"Validation loss: {test_loss/len(testloader):.3f}.. "
          f"Validation accuracy: {accuracy/len(testloader):.3f}")

wandb.finish()
```
