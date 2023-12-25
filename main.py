import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from Model import Model

data_dir = './Dataset/data/'
num_classes = len(os.listdir(data_dir))
model = Model(num_classes=num_classes)

train_dir = './Dataset/data'

# Load data using ImageFolder
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = ImageFolder(root=train_dir, transform=transform)

# Create DataLoader
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Define your training function
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


# Train the model
train_model(model, train_loader, epochs=10, learning_rate=0.001)
