# Traffic Vehicle Classification

A deep learning project for multi-class traffic vehicle image classification using PyTorch.

The project started as an experiment-driven image classification pipeline, progressing from a custom CNN baseline to transfer learning with ResNet architectures. The project was later extended with a Streamlit-based inference application and, in version `V3`, with a Vision Transformer (ViT) model and Energy-based Out-of-Distribution (OOD) detection.

---

# Overview

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
* Evaluate Vision Transformer based classification
* Maintain explicit experiment configurations
* Separate training, evaluation, and inference pipelines
* Preserve trained models as reusable artifacts
* Provide a practical inference application
* Support single-image and batch prediction
* Investigate unknown-class detection using OOD methods
* Provide human-review warnings for suspicious predictions

---

# Project Structure

```text
Traffic-Vehicle-Classification/
│
├── app.py
│
├── configs/
│   ├── experiments.yaml
│   ├── resnet_experiments.yaml
│   └── vit_experiments.yaml
│
├── dataset/
│
├── dataset_vit/
│
├── prediction/
│   ├── __init__.py
│   └── predictor.py
│
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   └── training/
│
├── ARTIFACTS/
│   ├── baseline/
│   ├── resnet/
│   └── vit/
│
├── tests/
│
├── pyproject.toml
├── uv.lock
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

The ViT experiments use a dedicated dataset configuration:

```text
dataset_vit/
```

The ViT dataset contains 3609 images with an 80/20 training-validation split:

```text
Training:   2888 images
Validation:  721 images
Test:       0 images
```

The image resolution used for the ViT experiments is:

```text
224 × 224
```

Image normalization follows the standard ImageNet normalization used by the pretrained Vision Transformer.

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

ResNet is one of the main families of architectures investigated in the project.

The project evaluates transfer learning and fine-tuning strategies using ResNet-based models.

The experiments investigate questions such as:

1. How useful are pretrained visual representations for this dataset?
2. Does fine-tuning improve over feature extraction?
3. How does the depth of fine-tuning affect performance?
4. How do different ResNet configurations compare?

The trained models are preserved as reusable checkpoints and can be selected directly from the inference application.

The project currently contains ResNet18 and ResNet34 experiments with different fine-tuning strategies.

---

# Vision Transformer

Version `V3` introduces a Vision Transformer model into the existing model ecosystem.

The selected architecture is:

```text
ViT-B/16
```

using the torchvision implementation with pretrained weights.

The ViT experiment was designed as a direct extension of the existing CNN and ResNet pipeline rather than as a separate project.

The main configuration of the first ViT experiment is:

```text
Architecture:     ViT-B/16
Pretrained:       True
Strategy:         Feature Extraction
Batch Size:       16
Optimizer:        AdamW
Learning Rate:    0.001
Weight Decay:     0.0001
Scheduler:        ReduceLROnPlateau
Epochs:           20
Image Size:       224 × 224
Seed:             42
```

The best checkpoint was obtained at epoch 12.

Validation performance:

```text
Validation Accuracy: 0.9584
Macro F1:            0.9570
ROC-AUC:             0.9968
```

The trained checkpoint is preserved as:

```text
ARTIFACTS/
└── vit/
    └── V1/
        └── best_model.pt
```

The ViT model is integrated into the same model-selection and prediction pipeline used by the CNN and ResNet models.

---

# Model Selection

The inference application provides multiple trained models that can be selected by the user.

The current model ecosystem contains:

```text
Baseline CNN
ResNet18
ResNet34
ViT-B/16
```

The model registry is responsible for connecting a model name with its checkpoint, architecture, and additional inference configuration.

The ViT model also contains an Energy threshold used for OOD analysis.

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
* Weighted F1
* Validation Loss
* ROC-AUC

Macro-averaged metrics are particularly important because they give each class equal importance regardless of the number of samples.

Confusion matrices are also used to identify class-level weaknesses and visually similar vehicle categories.

For the ViT experiments, OOD-related analysis is additionally performed using Energy scores.

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
9. Calculating the Energy score when OOD detection is available
10. Comparing the Energy score against the configured OOD threshold

Conceptually:

```text
Input Image
     ↓
Preprocessing
     ↓
Selected Model
     ↓
Logits
     ↓
Class Probabilities
     ↓
Predicted Class
     ↓
Confidence
     ↓
OOD Analysis
     ↓
Human Review Warning
```

The application displays the predicted class together with its confidence.

Example:

```text
Prediction: taxi
Confidence: 94.3%
```

For models with Energy-based OOD detection, additional information is available:

```text
Raw Prediction: vanet
Confidence: 91.2%
Energy Score: ...
Energy Threshold: ...
```

The classifier prediction and OOD detection are treated as two different signals.

The classifier answers:

```text
"If I must choose one of the known classes, which class is the most likely?"
```

The OOD detector answers:

```text
"Does this input look sufficiently similar to the known training distribution?"
```

Therefore, an image can have a high classification confidence while still being flagged as potentially outside the known distribution.

---

# Energy-Based OOD Detection

Version `V3` introduces Energy-based OOD detection for the ViT model.

The Energy score is calculated from the model logits using:

```text
Energy(x) = -logsumexp(logits)
```

The interpretation used in this project is:

```text
More negative Energy
        ↓
Stronger evidence of belonging to the known distribution

Higher Energy
        ↓
More suspicious / potentially unknown
```

The current ViT threshold was calibrated using the 95th percentile of the known validation Energy distribution:

```text
Energy Threshold: -3.427922
```

The current decision rule is:

```text
Energy <= -3.427922
        ↓
Known

Energy > -3.427922
        ↓
Potentially Unknown
```

The calibration statistics from the known validation set are:

```text
Minimum:  -13.473774
Mean:      -6.846153
Median:    -6.770739
P90:       -3.959971
P95:       -3.427922
P99:       -2.415662
Maximum:   -1.438292
```

The threshold is considered a preliminary calibration rather than a final OOD benchmark.

Because the threshold is based on the 95th percentile of known validation samples, approximately 5% of known validation samples are expected to fall above the threshold by construction.

Future experiments will evaluate the threshold using dedicated OOD datasets rather than changing it only from individual observations.

---

# OOD and Human Review

The OOD detector is currently integrated into the Single Image prediction workflow.

The system does not treat `UNKNOWN` as a ninth vehicle class.

Instead, the original classifier prediction remains available and the OOD detector provides an additional warning.

For example:

```text
Raw Prediction: vanet
Confidence: 91.2%

Energy Analysis:
Potentially Unknown

⚠️ Possible Unknown Class — Please review manually.
```

This distinction is important because the OOD detector does not determine which known class the image belongs to.

It only determines whether the input appears sufficiently consistent with the known data distribution.

The current V3 behavior is therefore:

```text
                ┌──────────────────┐
Input Image ───►│   ViT Classifier │
                └────────┬─────────┘
                         │
                         ▼
                  Raw Prediction
                         │
                         ├───────────────┐
                         │               │
                         ▼               ▼
                    Confidence       Energy Score
                                         │
                                         ▼
                                  OOD Threshold
                                         │
                                  ┌──────┴──────┐
                                  │             │
                                Known       Suspicious
                                  │             │
                                  │             ▼
                                  │      Human Review
                                  │
                                  └──────────────┘
```

At the current stage, this OOD warning is implemented for Single Image prediction.

Batch-level OOD handling is reserved for a future version.

---

# Web Application

The Streamlit-based inference interface provides three main modes:

```text
Single Image
     │
     ├── Batch Prediction
     │
     └── Evaluation
```

The interface also allows the user to select between the available trained models.

The Single Image mode currently provides:

* Model selection
* Image upload
* Predicted class
* Confidence
* Raw prediction
* Energy score
* Energy threshold
* OOD warning when the input is suspicious

Batch Prediction and Evaluation remain available for the existing classification workflow.

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
     ↓
Energy Analysis
     ↓
Possible OOD Warning
```

For models that support Energy-based OOD detection, the application also displays the raw prediction and Energy analysis.

Example:

```text
Raw Prediction: vanet
Confidence: 91.2%
Energy Score: ...
Energy Threshold: -3.427922
```

If the Energy score is above the configured threshold, the application displays:

```text
⚠️ Possible Unknown Class — Please review manually.
```

The classifier prediction is still displayed. The OOD detector acts as a warning layer rather than replacing the classifier output.

---

## 4. Batch Prediction

The batch prediction mode is intended for processing multiple images.

The current batch workflow focuses on classification and prediction output.

Conceptually:

```text
Input Dataset
     ↓
Select Model
     ↓
Batch Inference
     ↓
Predictions
     ↓
Confidence
     ↓
Organized Results
```

Batch-level OOD detection is not part of the current V3 scope.

It is planned for a future version after the Single Image OOD workflow has been evaluated further.

---

# Application Workflow

The overall application workflow is:

```text
                         Streamlit App
                              │
                   ┌──────────┴──────────┐
                   │                     │
             Select Model          Prediction Mode
                   │                     │
       ┌───────────┼───────────┐   ┌─────┼──────┐
       │           │           │   │     │      │
    CNN/ResNet    ViT      Other   Single Batch Evaluation
       │           │           │   │
       └───────────┴───────────┘   │
                   │               │
                   └───────┬───────┘
                           ↓
                       Prediction
                           ↓
                  Class + Confidence
                           ↓
                    Energy Analysis
                           ↓
                  Possible OOD Warning
```

The OOD analysis is currently associated with models that have an Energy threshold configured in the model registry.

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
* Distribution-shifted inputs
* Potentially unknown objects

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

OOD analysis adds another perspective by asking whether an input belongs to the same general distribution as the known training data.

A practical distribution-shift test was also considered using Nissan vehicle images.

However, such images should not be treated as a clean semantic OOD benchmark because a Nissan pickup can still be visually similar to one of the known vehicle categories, particularly `vanet`.

Therefore, stronger future OOD evaluation should include clearly unrelated classes such as:

```text
motorcycles
bicycles
animals
buildings
unrelated objects
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

The ViT experiments are configured separately through:

```text
configs/vit_experiments.yaml
```

The trained ViT checkpoint is preserved under:

```text
ARTIFACTS/vit/V1/
```

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
* ViT experiments
* Error analysis
* OOD detection
* Energy score calibration

The README provides the high-level project overview and application usage, while the reports contain the detailed methodology, experiment configurations, metrics, and analysis.

---

# Version 3

Version `V3` extends the existing classification and inference pipeline with Vision Transformer classification and Energy-based OOD detection.

The evolution of the project is:

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
├── Batch Prediction
└── Evaluation
        │
        ▼
V3
│
├── Vision Transformer
├── ViT-B/16
├── Model Registry Integration
├── Energy-based OOD Detection
├── Energy Threshold Calibration
└── Human Review Warning
```

The main focus of `V3` is moving from a conventional closed-set image classification system toward a classification system that can also identify suspicious inputs that may not belong to the known data distribution.

The current V3 OOD workflow is intentionally limited to Single Image prediction.

Batch OOD detection is reserved for a future version.

---

# Future Work

## V4 — Extended OOD and Model Analysis

The next major development stage can extend the current OOD system beyond the initial Energy-based implementation.

Potential directions include:

* Batch OOD Detection
* Dedicated OOD Evaluation Dataset
* MSP
* Temperature Scaling
* Mahalanobis Distance
* Comparison of multiple OOD methods
* Threshold calibration using known and unknown validation sets
* ViT fine-tuning
* Improved error analysis
* Cascade / Hierarchical Classification
* Grad-CAM and explainability
* Inference optimization
* API deployment
* Production-oriented model serving

The main goal of the next stage is not simply adding more techniques, but determining which OOD strategy provides the most useful behavior for this specific dataset and application.

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
        ├── Model Selection
        ├── Single Prediction
        ├── Batch Prediction
        └── Evaluation
        │
        ▼
V3
│
└── ViT & OOD Detection
        │
        ├── Vision Transformer
        ├── ViT-B/16
        ├── Energy Score
        ├── OOD Threshold
        └── Human Review Warning
        │
        ▼
V4
│
├── Batch OOD
├── Dedicated OOD Evaluation
├── OOD Method Comparison
├── Threshold Calibration
└── Further Model Analysis
        │
        ▼
Future
│
├── Cascade Classification
├── Explainability
├── Inference Optimization
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
Version: V3
Framework: PyTorch
Task: Multi-Class Image Classification
Models: CNN / ResNet18 / ResNet34 / ViT-B/16
OOD Method: Energy-based OOD Detection
Interface: Streamlit
```

The project has evolved from an experiment-focused image classification pipeline into a model comparison and inference system with an initial OOD detection layer.

Version `V3` represents the integration of Vision Transformer classification and Energy-based OOD detection while preserving the existing classification pipeline.
