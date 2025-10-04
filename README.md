# Project README

This repository contains two Jupyter notebooks (`q1.ipynb` and `q2.ipynb`) demonstrating deep learning models for image classification and image segmentation tasks.

---

## 1. q1.ipynb: Vision Transformer for CIFAR-10 Classification

**Description:**
Implements a Vision Transformer (ViT) architecture to classify images from the CIFAR-10 dataset into 10 classes.

**Key Features:**
- Data loading and preprocessing using `torchvision.datasets` and `transforms`.
- ViT components:
  - `PatchEmbedding` to split and embed image patches.
  - Multi-head `TransformerEncoderLayer` for sequence modeling.
  - Fully connected `MLPHead` for final classification.
- Training loop with configurable hyperparameters (learning rate, epochs, batch size).
- Evaluation on validation set and accuracy plotting.
- GPU acceleration support via PyTorch.

**Dependencies:**
- Python 3.8+
- torch
- torchvision
- matplotlib
- numpy

**Usage:**
1. Install dependencies: `pip install torch torchvision matplotlib numpy`
2. Run all cells in `q1.ipynb`.
3. Modify hyperparameters (learning rate, epochs, batch size) in the configuration cell as needed.


---

## 2. q2.ipynb: Text-Guided Segmentation with SAM and GroundingDINO

**Description:**
Performs zero-shot, text-prompted object segmentation on a sample image using:
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for region proposal based on text prompts.
- [Segment Anything Model (SAM) v1/v2](https://github.com/facebookresearch/segment-anything) for precise segmentation masks.

**Key Features:**
- Installs required libraries: `segment-anything`, `transformers`, `datasets`, `accelerate`, `opencv-python`, `Pillow`, `matplotlib`.
- Loads an image from a remote URL with `requests` and `Pillow`.
- Accepts a text prompt (e.g., "a dog") to specify the target object.
- Uses GroundingDINO to generate bounding box region seeds for the prompt.
- Feeds the region seeds to SAM predictor to produce a binary segmentation mask.
- Displays original image with overlayed segmentation mask.

**Dependencies:**
- Python 3.8+
- segment-anything
- transformers
- datasets
- accelerate
- opencv-python
- Pillow
- matplotlib
- torch

**Usage:**
1. Install dependencies:
   ```bash
   pip install segment-anything transformers datasets accelerate opencv-python pillow matplotlib torch
   ```
2. Open `q2.ipynb` and run all cells in order.
3. Input a custom text prompt in the designated prompt cell.
4. Review the generated segmentation mask overlay.

