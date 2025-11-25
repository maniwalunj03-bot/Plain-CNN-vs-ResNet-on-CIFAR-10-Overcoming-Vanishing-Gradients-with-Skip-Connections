# Plain-CNN-vs-ResNet-on-CIFAR-10-Overcoming-Vanishing-Gradients-with-Skip-Connections
This project compares ResNet50 trained from scratch vs pretrained ImageNet weights on CIFAR-10 using PyTorch. Includes training logs, accuracy tracking, evaluation, and insights showing how pretrained networks converge faster and achieve higher performance with fewer epochs.

📌 ResNet50 on CIFAR-10 – Pretrained vs From-Scratch (PyTorch)

📂 Project Overview

This project compares two training strategies of ResNet50 on the CIFAR-10 dataset:

1️⃣ Training ResNet50 from scratch

2️⃣ Fine-tuning a pretrained ResNet50 (ImageNet weights)

The goal is to analyze how transfer learning impacts convergence speed, performance, and generalization.

| Model            | Training Time | Final Accuracy | Observation                            |
| ---------------- | ------------- | -------------- | -------------------------------------- |
| **From Scratch** | Longer        | Lower          | Needs many epochs to learn features    |
| **Pretrained**   | Faster        | Higher         | Learns quickly using ImageNet features |

Using pretrained weights provides:

✔ Faster convergence

✔ Higher accuracy

✔ Better generalization

✔ Less overfitting

🧠 Why Transfer Learning Works

Instead of learning complex visual features (edges → textures → patterns) from zero, the pretrained ResNet50 already has useful filters.

Fine-tuning only adjusts them for CIFAR-10, making training efficient and effective.

📎 Project Features

Full training pipeline

Detailed logging of train/validation accuracy

Learning curves & performance visualization

Comparison between both models

Clean and well-commented PyTorch code

📁 Dataset

CIFAR-10 (10 image classes, 50k train, 10k test)

🛠 Tech Stack

Python

PyTorch

Torchvision

Matplotlib / Seaborn

CUDA (optional)

git clone <your_repo_link>
cd project_folder
pip install -r requirements.txt

Train from scratch:python train_scratch.py

Train pretrained:python train_pretrained.py

📊 Next Improvements

Grad-CAM interpretability

Confusion matrix visualizations

Experiment with different learning schedules

Compare multiple architectures (VGG, DenseNet, MobileNet)

🙋‍♀️ Author

Manisha Kalekar

Chemistry → Machine Learning | Deep Learning | Computer Vision
