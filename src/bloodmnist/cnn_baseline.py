# -*- coding: utf-8 -*-

import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from medmnist import BloodMNIST, INFO
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms


device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
batch_size = 64
epochs = 200
lr = 0.001

# Data Loading
data_flag = 'bloodmnist'
info = INFO[data_flag]
n_classes = len(info['label'])

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# --------- Before Training ----------
total_start = time.time()

#Model
class BloodMNISTCNN(nn.Module):
    def __init__(self, num_classes, with_softmax=False):
        super().__init__()
        self.with_softmax = with_softmax
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.feature_count = 128 * 28 * 28
        
        self.fc1 = nn.Linear(self.feature_count, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        #softmax 
        if self.with_softmax:
            x = F.softmax(x, dim=1)
            
        return x

#Training Function

def train_epoch(loader, model, criterion, optimizer):
    model.train()
    total_loss = 0.0
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        labels = labels.squeeze().long() # fix label shape

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

#Evaluation Function

def evaluate_accuracy(loader, model):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.squeeze().long()

            outputs = model(imgs)
            preds += outputs.argmax(dim=1).cpu().tolist()
            targets += labels.tolist()

    return accuracy_score(targets, preds)


def plot_metric(plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(list(range(1, len(plottable) + 1)), plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')

train_dataset = BloodMNIST(split='train', transform=transform, download=True, size=28)
val_dataset   = BloodMNIST(split='val',   transform=transform, download=True, size=28)
test_dataset  = BloodMNIST(split='test',  transform=transform, download=True, size=28)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# initialize the model
# with_softmax=False because CrossEntropyLoss expects logits
# to test "WITH Softmax" change to True 
model = BloodMNISTCNN(n_classes, with_softmax=False).to(device)

# get an optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# get a loss criterion
criterion = nn.CrossEntropyLoss()

# training loop
train_losses = []
val_accs = []
best_val_acc = 0.0
best_model_path = "best_bloodmnist_model.pth"

for epoch in range(epochs):

    epoch_start = time.time()

    train_loss = train_epoch(train_loader, model, criterion, optimizer)
    val_acc = evaluate_accuracy(val_loader, model)
    
    # Check if this is the best model so far
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # Save the state dictionary of the best model
        torch.save(model.state_dict(), best_model_path)
        print("New best model saved as best_bloodmnist_model.pth")


    train_losses.append(train_loss)
    val_accs.append(val_acc)

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | "
          f"Time: {epoch_time:.2f} sec")

# Test Accuracy of model with best perf on validation set
print(f"Loading best model with Val Acc: {best_val_acc:.4f}")
model.load_state_dict(torch.load(best_model_path, map_location=device))
test_acc = evaluate_accuracy(test_loader, model)
print("Best Model Test Accuracy:", test_acc)

# --------- After Training ----------
total_end = time.time()
total_time = total_end - total_start

print(f"\nTotal training time: {total_time/60:.2f} minutes "
      f"({total_time:.2f} seconds)")

plot_metric(train_losses, ylabel='Loss', name='CNN-baseline-training-loss')
plot_metric(val_accs, ylabel='Accuracy', name='CNN-baseline-validation-accuracy')