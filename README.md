# Traffic Vehicle Classification

A deep learning project for multi-class traffic vehicle image classification using PyTorch.

The project started as an experiment-driven image classification pipeline, progressing from a custom CNN baseline to transfer learning with ResNet architectures. In version `1.1.0`, the project is extended with a Streamlit-based inference application that supports single-image prediction and batch prediction for both labeled and unlabeled image datasets.

---

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
* Investigate training and optimization strategies
* Evaluate transfer learning with ResNet architectures
* Compare different model configurations
* Maintain explicit experiment configurations
* Separate training, evaluation, and inference pipelines
* Preserve trained models as reusable artifacts
* Provide a practical inference application
* Support single-image and batch prediction
* Support both labeled and unlabeled image datasets

---

# Project Structure

```text
Traffic-Vehicle-Classification/
│
├── app.py
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

The project is organized around reproducible experiments, trained model artifacts, evaluation, and a separate inference application.

---

# Dataset

The dataset contains eight vehicle classes:

| Class       |
| ----------- |
| `ambulance` |
| `autobus`   |
| `kamyun`    |
| `kamyunet`  |
| `minibus`   |
| `savari`    |
| `taxi`      |
| `vanet`     |

The dataset contains several visually similar vehicle categories, particularly:

```text
kamyun
kamyunet
vanet
```

and:

```text
autobus
minibus
```

Because of these similarities, aggregate accuracy alone is not sufficient to understand model behavior. Macro-averaged metrics, confusion matrices, and class-level error analysis are used throughout the project.

Before model training, the dataset was audited to verify its structure and image data.

---

# Models

## CNN Baseline

A custom CNN was trained from scratch to establish a quantitative reference before introducing transfer learning.

The architecture consists of four convolutional stages with increasing feature capacity:

```text
3 → 32 → 64 → 128 → 256
```

followed by global pooling and a fully connected classifier.

The CNN serves as a reference model for evaluating the benefit of more advanced architectures.

---

# ResNet

ResNet is the main family of architectures investigated in the project.

The project evaluates transfer learning and fine-tuning strategies using ResNet-based models.

The experiments investigate questions such as:

1. How useful are pretrained visual representations for this dataset?
2. Does fine-tuning improve over feature extraction?
3. How does the depth of fine-tuning affect performance?
4. How do different ResNet configurations compare?

The trained models are preserved as reusable checkpoints and can be selected directly from the inference application.

---

# Model Selection

The inference application currently provides four trained models that can be selected by the user.

The default model is:

```text
ResNet34-2
```

Additional models can be selected directly from the application without changing the prediction pipeline.

The project is designed so that new architectures can be added to the model selection system as the project evolves.

---

# Evaluation

Model evaluation is based on multiple metrics rather than accuracy alone.

The main metrics include:

* Validation Accuracy
* Macro Precision
* Macro Recall
* Macro F1
* Validation Loss

Macro-averaged metrics are particularly important because they give each class equal importance regardless of the number of samples.

Confusion matrices are also used to identify class-level weaknesses and visually similar vehicle categories.

---

# Prediction Pipeline

The prediction pipeline handles:

1. Loading the selected trained checkpoint
2. Restoring the model architecture
3. Loading class names
4. Applying the required preprocessing
5. Running inference
6. Calculating class probabilities
7. Returning the predicted class
8. Returning prediction confidence

Conceptually:

```text
Input Image
     ↓
Preprocessing
     ↓
Selected Model
     ↓
Class Probabilities
     ↓
Predicted Class
     ↓
Confidence
```

The application displays the predicted class together with its confidence.

Example:

```text
Prediction: taxi
Confidence: 94.3%
```

---

# Web Application

Version `1.1.0` introduces a Streamlit-based inference interface.

The application provides three main prediction modes:

```text
Single Image Prediction
        │
        ├── Batch Prediction — Labeled
        │
        └── Batch Prediction — Unlabeled
```

The interface also allows the user to select between the available trained models.

---

# Usage Guide

## 1. Installation

Clone the repository:

```bash
git clone <repository-url>
cd Traffic-Vehicle-Classification
```

Install the project dependencies using `uv`:

```bash
uv sync
```

---

## 2. Run the Application

Start the Streamlit application with:

```bash
uv run streamlit run app.py
```

After starting the application, Streamlit will provide a local URL that can be opened in a web browser.

---

## 3. Single Image Prediction

The single-image mode is designed for testing an individual image.

Typical workflow:

```text
Select Model
     ↓
Upload Image
     ↓
Run Prediction
     ↓
Predicted Class
     ↓
Confidence
```

The application returns the predicted vehicle class and its confidence score.

---

## 4. Batch Prediction — Labeled Dataset

The labeled batch mode is intended for datasets where the images are already organized according to their true classes.

The expected structure is:

```text
dataset/
├── ambulance/
├── autobus/
├── kamyun/
├── kamyunet/
├── minibus/
├── savari/
├── taxi/
└── vanet/
```

The application processes the images and compares the model predictions against the actual class labels.

This makes the labeled batch mode useful not only for inference, but also for evaluating a trained model on a collection of images.

The workflow is:

```text
Labeled Dataset
      ↓
Select Model
      ↓
Batch Inference
      ↓
Compare Prediction vs True Label
      ↓
Evaluation Results
      ↓
Organized Prediction Output
```

The predicted images are also organized into newly created class folders based on the model's predictions.

---

## 5. Batch Prediction — Unlabeled Dataset

The unlabeled batch mode is intended for images that do not have an existing class structure.

For example:

```text
images/
├── image_001.jpg
├── image_002.jpg
├── image_003.jpg
└── ...
```

The application classifies the images automatically.

The predicted images are then organized into newly created class folders:

```text
predictions/
├── ambulance/
├── autobus/
├── kamyun/
├── kamyunet/
├── minibus/
├── savari/
├── taxi/
└── vanet/
```

This makes it possible to use the application as an image-sorting tool in addition to a classification system.

The resulting prediction folders can be saved as a ZIP archive.

---

# Batch Prediction Output

Both labeled and unlabeled batch prediction modes organize the processed images according to their predicted class.

Conceptually:

```text
Input Dataset
     ↓
Batch Inference
     ↓
┌───────────────┐
│ Prediction    │
│ + Confidence  │
└───────────────┘
     ↓
Class-based Folders
     ↓
ZIP Archive
```

This allows the prediction results to be easily stored, transferred, or used for further analysis.

---

# Application Workflow

The overall application workflow is:

```text
                    Streamlit App
                         │
             ┌───────────┴───────────┐
             │                       │
       Select Model             Prediction Mode
             │                       │
       ┌─────┴─────┐        ┌───────┼─────────┐
       │           │        │       │         │
    Model 1     Model 2   Single   Labeled  Unlabeled
       │           │      Image    Batch      Batch
       └─────┬─────┘        │       │         │
             │              └───────┴─────────┘
             │                      │
             └──────────┬───────────┘
                        ↓
                    Prediction
                        ↓
                 Class + Confidence
                        ↓
                 Organized Results
```

---

# Error Analysis

Model performance is not evaluated only through aggregate metrics.

Class-level error analysis can be used to investigate:

* Visually similar vehicles
* Unusual viewpoints
* Low-light images
* Blurred images
* Occluded vehicles
* Unusual backgrounds
* Potentially ambiguous labels

Confusion matrices and representative incorrect predictions provide additional information about where the model struggles.

This is particularly important for visually similar classes such as:

```text
kamyun
kamyunet
vanet
```

and:

```text
autobus
minibus
```

---

# Reproducibility

The project uses explicit experiment configurations and a fixed random seed.

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

---

# Experiment Reports

Detailed technical analysis is kept separately from this README.

The project reports cover topics such as:

* Dataset audit
* CNN baseline experiments
* Ablation studies
* ResNet experiments
* Model comparison
* Error analysis

The README provides the high-level project overview and application usage, while the reports contain the detailed methodology, experiment configurations, metrics, and analysis.

---

# Version 1.1.0

Version `1.1.0` extends the original machine-learning pipeline into a usable inference application.

The main additions are:

```text
v1.0.0
│
├── Dataset
├── Data Audit
├── CNN Baseline
├── ResNet Experiments
├── Checkpoint Selection
└── Prediction Pipeline
        │
        ▼
v1.1.0
│
├── Streamlit Web UI
├── Model Selection
├── Single Image Prediction
├── Labeled Batch Prediction
├── Unlabeled Batch Prediction
├── Confidence Display
└── ZIP Export of Organized Predictions
```

The main focus of `v1.1.0` is moving from an experiment-focused ML pipeline toward a practical image classification application.

---

# Future Work

## v1.2.0 — Vision Transformer

The next major model expansion is planned around Vision Transformers (ViT).

The goal is to add ViT to the existing model ecosystem and compare its behavior against the current CNN and ResNet-based architectures.

The planned comparison will include:

```text
CNN
 │
 ├── Baseline CNN
 │
 └── ResNet
       │
       └── ResNet34-2
       
ViT
 │
 └── Vision Transformer
```

Future development may also include:

* OOD / Unknown Class Detection
* MSP
* Temperature Scaling
* OpenMax
* Cascade / Hierarchical Classification
* Grad-CAM and explainability
* Improved error analysis
* Inference optimization
* API deployment
* Production-oriented model serving

---

# Roadmap

```text
v1.0.0
│
└── ML Pipeline & Model Evaluation
        │
        ▼
v1.1.0
│
└── Application & Inference
        │
        ├── Streamlit UI
        ├── Single Prediction
        ├── Labeled Batch Prediction
        ├── Unlabeled Batch Prediction
        └── ZIP Export
        │
        ▼
v1.2.0
│
└── ViT & Model Expansion
        │
        ▼
Future
│
├── OOD Detection
├── Cascade Classification
├── Explainability
└── Deployment
```

---

# Technology Stack

```text
Python
PyTorch
Torchvision
Streamlit
uv
YAML
NumPy
Pandas
Scikit-learn
```

---

# Version

```text
Version: 1.1.0
Framework: PyTorch
Task: Multi-Class Image Classification
Primary Model: ResNet34-2
Interface: Streamlit
```

The `v1.0.0` tag represents the frozen first version of the ML classification pipeline.

The `v1.1.0` release adds the application and inference layer on top of that pipeline.
