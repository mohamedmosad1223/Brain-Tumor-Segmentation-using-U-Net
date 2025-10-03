# Brain-Tumor-Segmentation-using-U-Net
📌 Project Overview

This project implements a 3-layer U-Net model in PyTorch to perform brain tumor segmentation on the LGG MRI dataset.
The model learns to predict binary segmentation masks (tumor vs. non-tumor regions) from MRI scans.

📂 Dataset

Dataset: LGG MRI Segmentation
 (from Kaggle).

Each patient folder contains:

T1-weighted MRI images (.tif files).

Ground truth masks (_mask.tif files).

⚙️ Data Preprocessing

Convert images and masks to grayscale.

Normalize images to [0,1].

Masks are binarized (0 = background, 1 = tumor).

Data augmentation using Albumentations:

Resize to 256×256.

Random flips and rotations for training.

Validation only resized.

🏗️ Model Architecture

A simplified U-Net with 3 encoder–decoder levels:

Encoder:

Conv → BN → ReLU blocks (64, 128, 256).

Bottleneck:

Conv block (512).

Decoder:

Transposed convolutions for upsampling.

Skip connections from encoder.

Conv → BN → ReLU blocks (256, 128, 64).

Output:

1-channel conv layer for binary mask prediction.

🧪 Training Setup

Loss: BCEWithLogitsLoss (binary segmentation).

Optimizer: Adam (lr=1e-4).

Batch size: 8.

Epochs: 25.

Metrics:

Dice Score.

IoU (Intersection over Union).

📊 Results

After 25 epochs of training:

Training Loss: ~0.0117

Validation Loss: ~0.0109

Training Dice: ~0.717

Validation Dice: ~0.743

Training IoU: ~0.600

Validation IoU: ~0.628

✅ The model shows strong segmentation performance and generalizes well to validation data.

📈 Visualization

The script plots:

Input MRI image.

Corresponding ground truth mask.

You can also extend it to visualize:

Predicted mask vs Ground truth for validation samples.

🚀 How to Run

Clone the repository & download dataset from Kaggle.

Install requirements:

pip install torch torchvision albumentations scikit-learn matplotlib


Train the model:

python train.py


