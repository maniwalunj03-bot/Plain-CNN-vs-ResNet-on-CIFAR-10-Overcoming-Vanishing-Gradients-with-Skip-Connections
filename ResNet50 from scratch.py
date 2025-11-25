# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 12:04:23 2025

@author: Manisha
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used: ", device)

# CIFAR-10 Transforms
# Training: augmentations (crop, flip, normalize)
# Testing: only normalization
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transforms
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=test_transforms
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# Define ResNet Model (2B)
# We will use ResNet-18 pretrained on ImageNet, but modify the final layer for CIFAR-10.
model = models.resnet50(weights=None)   # TRAIN FROM SCRATCH
model.fc = nn.Linear(model.fc.in_features, 10)   # CIFAR-10 (10 classes)
model = model.to(device)

# Optimizer + Scheduler (3D)
# Use Adam + StepLR scheduler, stepping every 10 epochs.
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)

# Training + Validation Loop
import time

epochs = 30
best_acc = 0.0

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

for epoch in range(epochs):
    start_time = time.time()

    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100 * correct / total
    history["train_loss"].append(train_loss / len(train_loader))
    history["train_acc"].append(train_acc)


    # --------------------
    # Validation
    # --------------------
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = 100 * correct / total
    history["val_loss"].append(val_loss / len(test_loader))
    history["val_acc"].append(val_acc)

    # Step the LR scheduler
    scheduler.step()

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "resnet18_best.pth")

    print(f"Epoch [{epoch+1}/{epochs}]  "
          f"Train Loss: {train_loss/len(train_loader):.4f}  "
          f"Train Acc: {train_acc:.2f}%  "
          f"Val Loss: {val_loss/len(test_loader):.4f}  "
          f"Val Acc: {val_acc:.2f}%  "
          f"Time: {(time.time()-start_time):.1f}s")

    # Plot training curves
    import matplotlib.pyplot as plt

def plot_training_curves(history):
    train_loss = history["train_loss"]
    val_loss   = history["val_loss"]
    train_acc  = history["train_acc"]
    val_acc    = history["val_acc"]

    epochs = range(1, len(train_loss)+1)

    plt.figure(figsize=(14,5))

    # Loss curve
    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy curve
    plt.subplot(1,2,2)
    plt.plot(epochs, train_acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

plot_training_curves(history)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
# Confusion Matrix
def plot_confusion_matrix(model, test_loader, device, class_names):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            _, predicted = torch.max(output, 1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(y.cpu().numpy())

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(16,14))
    sns.heatmap(cm, annot=False, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

# Example:
plot_confusion_matrix(model, test_loader, device, class_names)

target_layers = [model.layer4[-1]]

def generate_gradcam(model, data_loader, class_names, device, num_images=5):

    cam = GradCAM(model=model, target_layers=target_layers)

    model.eval()
    images_done = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        grayscale_cam_batch = cam(input_tensor=images)

        for i in range(len(images)):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())

            grayscale_cam = grayscale_cam_batch[i]
            heatmap = show_cam_on_image(img, grayscale_cam, use_rgb=True)

            plt.figure(figsize=(4,4))
            plt.imshow(heatmap)
            plt.title(f"True: {class_names[labels[i].item()]}")
            plt.axis("off")
            plt.show()

            images_done += 1
            if images_done >= num_images:
                return
generate_gradcam(
    model=model,
    data_loader=test_loader,
    class_names=class_names,
    device=device,
    num_images=6    # show 6 images
)
import os

os.makedirs("gradcam_results", exist_ok=True)

def save_gradcam_images(model, data_loader, class_names, device, num_images=10):

    cam = GradCAM(model=model, target_layers=target_layers)

    model.eval()
    saved = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        grayscale_cam_batch = cam(input_tensor=images)

        for i in range(len(images)):

            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())

            grayscale_cam = grayscale_cam_batch[i]
            heatmap = show_cam_on_image(img, grayscale_cam, use_rgb=True)

            plt.imsave(
                f"gradcam_results/{saved}_{class_names[labels[i].item()]}.png",
                heatmap
            )

            saved += 1
            if saved >= num_images:
                print(f"Saved {num_images} Grad-CAM images to gradcam_results/")
                return
save_gradcam_images(model, test_loader, class_names, device)

