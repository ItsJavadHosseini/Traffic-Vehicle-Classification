# Traffic Vehicle Classification

A deep learning project for multi-class traffic vehicle image classification using PyTorch.

The project focuses on building a reproducible image-classification pipeline, starting with a CNN baseline and progressing to transfer learning with ImageNet-pretrained ResNet18.

## Overview

The goal of this project is to classify traffic vehicle images into eight vehicle categories:

* `ambulance`
* `autobus`
* `kamyun`
* `kamyunet`
* `minibus`
* `savari`
* `taxi`
* `vanet`

The project follows an experiment-driven approach rather than focusing only on final accuracy.

The main objectives are:

* Establish a reproducible CNN baseline
* Investigate the effect of training and optimization strategies
* Evaluate transfer learning with ResNet18
* Compare feature extraction and progressive fine-tuning
* Maintain explicit experiment configurations
* Separate training, evaluation, and prediction pipelines
* Preserve trained models as reusable artifacts

## Project Structure

```text
Traffic-Vehicle-Classification/
│
├── configs/
│   ├── baseline.yaml
│   ├── resnet.yaml
│   └── experiments.yaml
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── splits/
│
├── notebooks/
│   ├── 01_data_audit/
│   ├── 02_baseline/
│   ├── 03_ablations/
│   ├── 04_resnet/
│   └── 05_error_analysis/
│
├── prediction/
│   ├── __init__.py
│   ├── predictor.py
│   └── preprocessing.py
│
├── runs/
│
├── src/
│   ├── models/
│   ├── evaluation/
│   └── ...
│
├── tests/
│
├── ARTIFACTS/
│   ├── baseline/
│   └── resnet/
│
├── pyproject.toml
└── README.md
```

The project is organized around reproducible experiments, model artifacts, and a separate inference pipeline.

# Dataset

The dataset contains **3,982 images** distributed across eight vehicle classes.

| Class       |    Images |
| ----------- | --------: |
| `ambulance` |       350 |
| `autobus`   |       482 |
| `kamyun`    |       437 |
| `kamyunet`  |       609 |
| `minibus`   |       459 |
| `savari`    |       561 |
| `taxi`      |       533 |
| `vanet`     |       551 |
| **Total**   | **3,982** |

The dataset is moderately imbalanced, with the number of images per class ranging from 350 to 609.

Several vehicle categories are visually similar, particularly:

```text
kamyun
kamyunet
vanet
autobus
minibus
```

For this reason, aggregate accuracy alone is not sufficient to understand model behavior. Macro-averaged metrics, confusion matrices, and class-level error analysis are used throughout the project.

Before model training, the dataset was audited to verify its structure and image data.

# Models

## CNN Baseline

A custom CNN was trained from scratch to establish a quantitative reference before introducing transfer learning.

The architecture consists of four convolutional stages with increasing feature capacity:

```text
3 → 32 → 64 → 128 → 256
```

followed by global pooling and a fully connected classifier.

The selected baseline configuration uses:

```text
Dropout:        0.3
Pooling:        Max
Optimizer:      AdamW
Learning Rate:  0.001
Weight Decay:   0
Scheduler:      ReduceLROnPlateau
Augmentation:   Disabled
```

The best baseline configuration achieved:

```text
Validation Accuracy: 88.57%
Macro Precision:     88.84%
Macro Recall:        88.37%
Macro F1:            88.45%
Validation Loss:      0.3550
```

The CNN serves as the reference model for evaluating the benefit of pretrained architectures.

Detailed baseline experiments and ablation results are documented separately.

# ResNet18

ResNet18 is the main architecture investigated in this project.

The ResNet phase focuses on transfer learning and progressive fine-tuning using an ImageNet-pretrained ResNet18.

The original classification head is replaced with an eight-class classifier for the vehicle classification task.

Four experiments were designed:

* R1 — Feature Extraction
* R2 — Fine-Tuning Stage 1
* R3 — Fine-Tuning Stage 2
* R4 — Training from Scratch

## R1 — Feature Extraction

R1 uses ImageNet-pretrained ResNet18 primarily as a feature extractor.

The pretrained convolutional backbone remains frozen while the final classification layer is trained for the eight vehicle classes.

```text
ImageNet-pretrained ResNet18
            │
            ├── Frozen Backbone
            │
            └── Trainable FC
```

R1 establishes the initial transfer-learning reference.

## R2 — Fine-Tuning Stage 1

R2 continues from the best checkpoint obtained in R1.

The final ResNet block, `layer4`, is unfrozen together with the classification head.

Different learning rates are used for pretrained and newly initialized parameters:

```text
layer4 → 1e-4
fc     → 1e-3
```

The classification head receives larger updates because it is newly initialized, while `layer4` is adapted more conservatively from its pretrained representation.

This experiment evaluates whether adapting the highest-level convolutional features improves performance over pure feature extraction.

## R3 — Fine-Tuning Stage 2

R3 continues from the best checkpoint obtained in R2.

The trainable components are expanded to:

```text
layer3
layer4
fc
```

with progressively smaller learning rates:

```text
layer3 → 1e-5
layer4 → 1e-4
fc     → 1e-3
```

Conceptually:

```text
LR(layer3) < LR(layer4) < LR(fc)
```

This allows deeper pretrained representations to adapt while applying smaller updates to the more general features learned during ImageNet pretraining.

R3 represents the deepest fine-tuning stage in the V1 ResNet experiments.

## R4 — ResNet18 From Scratch

R4 removes ImageNet pretraining completely.

The entire ResNet18 is initialized randomly and trained using only the project dataset.

The configuration differs from the pretrained experiments:

```text
Initialization:   Random
Input Resolution: 64 × 64
Batch Size:       64
Epochs:           45
Scheduler:        CosineAnnealingLR
```

R4 provides an independent from-scratch reference for studying the contribution of pretrained representations.

Because R4 also uses a different input resolution and training configuration, its result should not be interpreted as a pure pretraining ablation against R1–R3.

# ResNet Experimental Design

The main transfer-learning progression is:

```text
                ResNet18
                   │
            ImageNet Pretrained
                   │
                  R1
                   │
          Fine-tune layer4
                   │
                  R2
                   │
      Fine-tune layer3 + layer4
                   │
                  R3
```

An independent branch evaluates training without pretrained weights:

```text
ResNet18
   │
Random Initialization
   │
   R4
```

The experiments investigate four main questions:

1. How useful are ImageNet-pretrained visual representations for this dataset?
2. Does fine-tuning `layer4` improve over feature extraction?
3. Does progressively deeper fine-tuning improve performance further?
4. How does a from-scratch ResNet18 compare with transfer learning?

# Training Configuration

The pretrained ResNet experiments use:

```text
Architecture:       ResNet18
Initialization:     ImageNet pretrained
Input Resolution:   224 × 224
Batch Size:         32
Optimizer:          AdamW
Weight Decay:       1e-4
Scheduler:          ReduceLROnPlateau
Epochs:             20
Normalization:      ImageNet
```

R4 uses:

```text
Architecture:       ResNet18
Initialization:     Random
Input Resolution:   64 × 64
Batch Size:         64
Optimizer:          AdamW
Weight Decay:       1e-4
Scheduler:          CosineAnnealingLR
Epochs:             45
```

The experiment parameters are defined through YAML configuration files rather than being hard-coded inside the training pipeline.

# Checkpoint Progression

The fine-tuning experiments are connected through model checkpoints:

```text
R1
 │
 └── Best Checkpoint
        ↓
       R2
        │
        └── Best Checkpoint
               ↓
              R3
```

R2 starts from the best R1 checkpoint.

R3 starts from the best R2 checkpoint.

This makes R1 → R2 → R3 a progressive fine-tuning pipeline rather than three completely independent training runs.

R4 is independent and starts from random initialization.

# Evaluation

Model evaluation is based on multiple metrics:

* Validation Accuracy
* Macro Precision
* Macro Recall
* Macro F1
* Validation Loss

Macro-averaged metrics are important because they give each class equal importance regardless of its number of samples.

Confusion matrices are also used to identify class-level weaknesses and visually similar vehicle categories.

The main difficult categories include:

```text
kamyun
kamyunet
vanet
autobus
minibus
```

# Prediction Pipeline

The trained ResNet model can be loaded independently from the training pipeline.

The prediction module handles:

1. Loading the trained checkpoint
2. Restoring the model architecture
3. Loading class names
4. Applying the required preprocessing
5. Running inference
6. Calculating class probabilities
7. Returning the predicted class and confidence

The ResNet inference pipeline is:

```text
Input Image
    ↓
RGB Conversion
    ↓
Resize
    ↓
Center Crop
    ↓
ToTensor
    ↓
ImageNet Normalization
    ↓
ResNet18
    ↓
Softmax
    ↓
Class + Confidence
```

Example output:

```text
Class: ambulance
Index: 0
Confidence: 1.0000
```

The prediction pipeline has been successfully tested with the trained ResNet checkpoint.

# Error Analysis

Model performance is not evaluated only through aggregate metrics.

After selecting the strongest ResNet configuration, class-level error analysis can be used to investigate:

* Visually similar vehicles
* Unusual viewpoints
* Low-light images
* Blurred images
* Occluded vehicles
* Unusual backgrounds
* Potentially ambiguous labels

Confusion matrices and representative incorrect predictions provide additional information about where the model struggles.

This is particularly important for visually similar classes such as `kamyun`, `kamyunet`, and `vanet`.

# Reproducibility

The project uses explicit experiment configurations and a fixed random seed:

```yaml
seed: 42
```

Important experimental parameters are defined explicitly, including:

* Dataset split
* Model architecture
* Initialization strategy
* Fine-tuning strategy
* Image resolution
* Batch size
* Optimizer
* Learning rates
* Weight decay
* Scheduler
* Epoch count
* Checkpoint source

This configuration-driven approach makes experiments easier to reproduce and compare.

# Experiment Reports

Detailed technical analysis is kept separately from this README.

The project reports cover:

* Dataset audit
* CNN baseline experiments
* Ablation studies
* ResNet18 experiments
* Error analysis

The README provides the high-level project overview, while the reports contain the detailed methodology, experiment configurations, metrics, and analysis.

# V1 Scope

Version `1.0.0` establishes the first complete version of the classification pipeline:

```text
Dataset
   ↓
Data Audit
   ↓
CNN Baseline
   ↓
ResNet18 Experiments
   ↓
Checkpoint Selection
   ↓
Prediction Pipeline
```

The primary focus of V1 is the development and evaluation of a ResNet18-based classification pipeline, supported by a from-scratch CNN baseline.

# Future Work

Potential directions for the next development phase include:

* MobileNet experiments
* ResNet34 experiments
* Streamlit inference interface
* Targeted data augmentation
* Deeper error analysis
* Confusion-matrix-driven dataset analysis
* Grad-CAM visualization
* Model comparison in the prediction interface
* Inference optimization and deployment

# Version

```text
Version: 1.0.0
Framework: PyTorch
Task: Multi-Class Image Classification
Primary Architecture: ResNet18
```

The `v1.0.0` tag represents the frozen first version of the project.
