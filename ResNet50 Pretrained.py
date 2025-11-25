# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 12:14:51 2025

@author: Manisha
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ===== DATASET =====
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=2)

# ===== PRETRAINED RESNET50 =====
model = models.resnet50(pretrained=True)

# Replace the last FC layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # CIFAR10 has 10 classes
model = model.to(device)

# ===== LOSS + OPTIMIZER =====
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)

scheduler = CosineAnnealingLR(optimizer, T_max=20)

# ===== TRAINING LOOP =====
def train_model(model, train_loader, test_loader, epochs=20):
    history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
    }
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()

        scheduler.step()

        train_acc = 100 * correct / len(train_loader.dataset)
        train_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_correct, val_loss = 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()

        val_acc = 100 * val_correct / len(test_loader.dataset)
        val_loss = val_loss / len(test_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch}/{epochs}]  "
            f"Train Acc: {train_acc:.2f}%  Val Acc: {val_acc:.2f}%"
        )

    return model
model = train_model(model, train_loader, test_loader, epochs=20)

torch.save(model.state_dict(), "resnet50_cifar10_pretrained.pth")
print("Model saved!")

import matplotlib.pyplot as plt

def plot_curves(history):
    """
    history is a dict like:
    {
        "train_loss": [...],
        "val_loss": [...],
        "train_acc": [...],
        "val_acc": [...]
    }
    """

    epochs = range(1, len(history["train_loss"]) + 1)

    # ----- LOSS CURVE -----
    plt.figure(figsize=(7,5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ----- ACCURACY CURVE -----
    plt.figure(figsize=(7,5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_curves(history)
    
# 1️ Load Pretrained ResNet50
# ---------------------------------------
model = models.resnet50(pretrained=True)
model.eval()

# We store gradients & activations here
gradients = None
activations = None

# ---------------------------------------
# 2️ Hook to capture gradients
# ---------------------------------------
def save_gradient(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def save_activation(module, input, output):
    global activations
    activations = output

# Register hooks on last conv layer of ResNet50
target_layer = model.layer4[-1].conv3
target_layer.register_forward_hook(save_activation)
target_layer.register_backward_hook(save_gradient)

# ---------------------------------------
# 3️ Preprocessing (ImageNet normalization)
# ---------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------------------
# 4️ Load input image
# ---------------------------------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    tensor_img = preprocess(img).unsqueeze(0)
    return img, tensor_img

# ---------------------------------------
# 5️ Grad-CAM computation
# ---------------------------------------
def generate_gradcam(tensor_img, class_idx=None):
    global gradients, activations

    # Forward pass
    output = model(tensor_img)

    # Get predicted class if not provided
    if class_idx is None:
        class_idx = output.argmax().item()

    # Backward pass on target class
    model.zero_grad()
    loss = output[0, class_idx]
    loss.backward()

    # Convert tensors to numpy
    grads = gradients.detach().cpu().numpy()
    acts = activations.detach().cpu().numpy()

    # Compute Global Average Pooling
    weights = grads.mean(axis=(2, 3))[0]

    # Weighted sum
    cam = np.zeros(acts.shape[2:], dtype=np.float32)
    for w, act in zip(weights, acts[0]):
        cam += w * act

    # ReLU
    cam = np.maximum(cam, 0)

    # Normalize for visualization
    cam -= cam.min()
    cam /= cam.max()

    return cam

# ---------------------------------------
# 6️⃣ Overlay Grad-CAM heatmap
# ---------------------------------------
def overlay_cam(img, cam):
    cam_resized = cv2.resize(cam, (img.width, img.height))
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    combined = np.float32(heatmap) / 255 + np.float32(img) / 255
    combined = combined / combined.max()
    return heatmap, combined

# ---------------------------------------
# 7️⃣ Complete Pipeline
# ---------------------------------------
def gradcam_pipeline(image_path, class_idx=None):
    img, tensor = load_image(image_path)
    cam = generate_gradcam(tensor, class_idx)
    heatmap, result = overlay_cam(img, cam)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Heatmap (Grad-CAM)")
    plt.imshow(heatmap)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay Result")
    plt.imshow(result)
    plt.axis("off")

    plt.show()

# ---------------------------------------
# 8️⃣ Run example
# ---------------------------------------
if __name__ == "__main__":
    image_path = "/content/sample.jpg"   # Change to your path
    gradcam_pipeline(image_path)
