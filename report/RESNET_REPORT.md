# ResNet Experiments Report

## 1. Overview

This phase evaluates the performance and training behavior of the ResNet18 architecture for the traffic vehicle classification task.

The main objective is not only to obtain a higher validation accuracy, but also to investigate how different transfer-learning strategies affect model performance.

Four experiments were designed:

* **R1 — Feature Extraction**
* **R2 — Fine-Tuning Stage 1**
* **R3 — Fine-Tuning Stage 2**
* **R4 — Training ResNet18 from Scratch**

The experiments progressively move from using ResNet18 as a fixed pretrained feature extractor toward deeper fine-tuning and finally a completely non-pretrained model.

This structure allows the contribution of pretrained representations and the effect of progressively unfreezing deeper layers to be evaluated in a controlled manner.

---

## 2. Dataset and Data Split

The experiments use the project's traffic vehicle classification dataset.

The dataset contains eight vehicle classes:

* ambulance
* autobus
* kamyun
* kamyunet
* minibus
* savari
* taxi
* vanet

A fixed random seed of **42** is used to improve reproducibility.

The dataset is divided into:

* **Training set:** 80%
* **Validation set:** 20%
* **Test set:** not used in this phase

The test split is intentionally set to zero because the current ResNet experiments are focused on model development and validation rather than final evaluation on an unseen test set.

```yaml
seed: 42

split:
  val_size: 0.2
  test_size: 0.0
```

---

## 3. ResNet18 Architecture

ResNet18 is used as the main pretrained architecture in this phase.

ResNet18 is a convolutional neural network based on residual learning. Instead of requiring every layer to directly learn a complete transformation, residual blocks allow layers to learn a residual function relative to their input.

Conceptually, a residual block can be represented as:

$$
y = F(x) + x
$$

where:

* \(x\) is the input
* \(F(x)\) is the transformation learned by the convolutional layers
* \(y\) is the output

The residual connection helps information and gradients propagate through the network and makes deeper architectures easier to optimize.

For the pretrained experiments, the ImageNet-pretrained ResNet18 is used.

The original classification head is adapted to the project's eight vehicle classes.

---

# 4. Experimental Strategy

The ResNet experiments are structured as a progressive transfer-learning pipeline.

### R1: Feature Extraction

The pretrained ResNet18 is initially treated primarily as a feature extractor.

Most of the pretrained convolutional representation remains frozen, while the classification head is trained for the vehicle classification task.

This experiment establishes a reference point for the usefulness of ImageNet-pretrained visual features without substantially modifying the pretrained backbone.

---

### R2: Fine-Tuning Stage 1

The second experiment starts from the best checkpoint obtained in R1.

Instead of keeping the entire pretrained backbone frozen, the final ResNet block (`layer4`) is unfrozen.

The classification head receives a larger learning rate, while `layer4` receives a smaller learning rate.

```yaml
parameter_groups:
  layer4:
    lr: 1.0e-4

  fc:
    lr: 1.0e-3
```

This creates a differential learning-rate strategy.

The classification head is randomly initialized for the new eight-class task and therefore requires relatively larger parameter updates.

In contrast, `layer4` already contains useful pretrained representations, so its parameters are updated more conservatively.

The experiment therefore tests whether adapting the highest-level pretrained features to the vehicle domain improves performance.

---

### R3: Fine-Tuning Stage 2

The third experiment continues from the best checkpoint of R2.

At this stage, another deeper portion of the network is allowed to adapt.

The trainable layers are:

* `layer3`
* `layer4`
* `fc`

with the following learning rates:

```yaml
parameter_groups:
  layer3:
    lr: 1.0e-5

  layer4:
    lr: 1.0e-4

  fc:
    lr: 1.0e-3
```

This creates a three-level learning-rate hierarchy.

$$
LR_{layer3} < LR_{layer4} < LR_{fc}
$$

The reasoning is that earlier pretrained representations are more general and should therefore be modified more cautiously, while the newly initialized classification head requires the largest updates.

R3 consequently tests whether adapting deeper feature representations provides additional benefit over R2.

---

### R4: ResNet18 From Scratch

The fourth experiment removes ImageNet pretraining completely.

```yaml
pretrained: false
```

The entire ResNet18 is initialized from scratch and trained using only the project's vehicle dataset.

This experiment provides an important comparison against transfer learning.

If R1–R3 outperform R4, this provides evidence that pretrained visual representations are beneficial for this dataset and training setup.

R4 therefore acts as a control experiment for evaluating the contribution of ImageNet pretraining.

---

# 5. Input Resolution

The pretrained experiments R1–R3 use an input resolution of:

$$
224 \times 224
$$

This is the standard input resolution associated with the ImageNet-pretrained ResNet18 configuration.

```yaml
image_size: 224
```

R4 intentionally uses a smaller input resolution:

$$
64 \times 64
$$

```yaml
image_size: 64
```

The smaller input size reduces computational cost and allows the from-scratch experiment to investigate whether ResNet18 can learn useful representations from the available dataset under a lower-resolution configuration.

Because R4 differs from R1–R3 in both pretraining and input resolution, its result should not be interpreted as a pure test of pretraining alone. It represents a separate training configuration and should be analyzed accordingly.

---

# 6. Normalization

The pretrained experiments use ImageNet normalization:

```yaml
normalization: "imagenet"
```

This is consistent with the statistics used during ImageNet pretraining and allows the input distribution to remain compatible with the pretrained feature representations.

R4 also uses the ImageNet normalization configuration despite being trained from scratch.

This keeps the normalization strategy consistent across the experiments while changing the initialization strategy.

---

# 7. Optimization Configuration

AdamW is used as the optimizer throughout all four ResNet experiments.

```yaml
optimizer:
  name: "AdamW"
```

The weight decay is fixed at:

$$
10^{-4}
$$

for all experiments.

AdamW separates weight decay from the gradient-based parameter update, providing a more direct implementation of decoupled weight regularization.

For R1 and R4, a global learning rate of:

$$
10^{-3}
$$

is used.

For R2 and R3, parameter-specific learning rates are used instead.

This distinction is important because fine-tuning pretrained networks generally requires smaller updates to pretrained parameters than to a newly initialized classification head.

---

# 8. Learning-Rate Strategies

Two different learning-rate scheduling strategies are used.

### R1–R3: ReduceLROnPlateau

The fine-tuning experiments use:

```yaml
scheduler:
  name: "ReduceLROnPlateau"
```

This scheduler responds to validation performance rather than reducing the learning rate purely according to epoch count.

The underlying idea is:

> If validation performance stops improving, reduce the learning rate so that optimization can continue with smaller parameter updates.

This is particularly appropriate for fine-tuning because the pretrained model may initially make rapid progress and later require smaller updates to adapt without excessively modifying useful pretrained representations.

---

### R4: CosineAnnealingLR

The from-scratch experiment uses:

```yaml
scheduler:
  name: "CosineAnnealingLR"
```

with 45 training epochs.

Cosine annealing gradually reduces the learning rate following a cosine-shaped schedule.

The purpose is to provide a smooth learning-rate decay throughout the longer from-scratch training process.

---

# 9. Training Configuration

The first three experiments each use:

$$
20 \text{ epochs}
$$

with a batch size of:

$$
32
$$

R4 uses:

$$
45 \text{ epochs}
$$

with a batch size of:

$$
64
$$

The larger number of epochs in R4 reflects the fact that the network must learn visual representations from scratch rather than starting from ImageNet-pretrained features.

---

# 10. Checkpoint Progression

The experiments are intentionally connected through checkpoints.

The progression is:

```text
R1
 │
 └── best checkpoint
       ↓
      R2
       │
       └── best checkpoint
             ↓
            R3
```

R1 starts without a checkpoint:

```yaml
checkpoint:
  load_from: null
```

R2 loads the best checkpoint from R1:

```yaml
checkpoint:
  load_from: "R1"
  load_best: true
```

R3 then loads the best checkpoint from R2:

```yaml
checkpoint:
  load_from: "R2"
  load_best: true
```

This means R2 is not an independent experiment initialized from the original pretrained model. It represents the next stage of adaptation after R1.

Similarly, R3 continues the fine-tuning process from R2.

R4 is independent and starts from scratch.

---

# 11. Experimental Matrix

| Experiment | Initialization        | Trainable Components |   Input | Batch | LR Strategy        | Epochs |
| ---------- | --------------------- | -------------------- | ------: | ----: | ------------------ | -----: |
| R1         | ImageNet pretrained   | Classification head  | 224×224 |    32 | 1e-3               |     20 |
| R2         | Best R1 checkpoint    | layer4 + FC          | 224×224 |    32 | 1e-4 / 1e-3        |     20 |
| R3         | Best R2 checkpoint    | layer3 + layer4 + FC | 224×224 |    32 | 1e-5 / 1e-4 / 1e-3 |     20 |
| R4         | Random initialization | Entire network       |   64×64 |    64 | 1e-3 + cosine      |     45 |

---

# 12. Main Experimental Questions

The experiment design is intended to answer several specific questions.

### Question 1 — Does ImageNet pretraining help?

R1 provides the first measurement of how effectively pretrained ResNet18 representations transfer to the vehicle classification problem.

R4 provides a from-scratch reference configuration.

The comparison should be interpreted together with the difference in input resolution between the two experiments.

---

### Question 2 — Does fine-tuning the final ResNet block improve performance?

R2 tests whether allowing `layer4` to adapt to the vehicle dataset improves upon the feature-extraction configuration in R1.

The experiment isolates the adaptation of the highest-level convolutional representation.

---

### Question 3 — Does deeper fine-tuning provide additional benefit?

R3 extends R2 by unfreezing `layer3`.

This tests whether the vehicle classification task benefits from adapting more of the pretrained representation.

The progressively smaller learning rate assigned to `layer3` is intended to reduce the risk of excessively modifying the more general pretrained features.

---

### Question 4 — How does a completely from-scratch ResNet compare?

R4 removes ImageNet initialization.

This experiment determines whether the available vehicle dataset contains enough information for ResNet18 to learn an effective representation without external visual pretraining.

---

# 13. Evaluation Metrics

The ResNet experiments should be evaluated using more than validation accuracy.

The primary metrics are:

* Validation Accuracy
* Macro Precision
* Macro Recall
* Macro F1
* Validation Loss

Macro-averaged metrics are particularly important because they give each vehicle class equal weight regardless of its number of samples.

For example:

$$
MacroF1 =
\frac{1}{C}
\sum_{c=1}^{C}F1_c
$$

where \(C=8\) represents the number of vehicle classes.

Confusion matrices should also be examined to determine which vehicle categories remain difficult to distinguish.

---

# 14. Error Analysis

After selecting the strongest ResNet configuration based on the validation results, error analysis should be performed at the class level.

Particular attention should be given to visually similar vehicle categories, such as:

* `kamyun`
* `kamyunet`
* `vanet`
* `autobus`
* `minibus`

The confusion matrix can reveal whether the model's remaining errors are concentrated around specific visually similar classes.

Incorrect predictions should then be inspected individually.

Useful categories for error analysis include:

* visually similar vehicles
* unusual viewpoints
* low-light images
* blurred images
* occluded vehicles
* unusual backgrounds
* incorrect or ambiguous labels

This analysis is important because a higher aggregate accuracy does not necessarily mean that all vehicle classes have improved equally.

---

# 15. Interpretation Framework

The results should be interpreted as a progression rather than as four completely unrelated models.

The intended progression is:

```text
R1
Feature Extraction
      ↓
R2
Fine-Tune layer4
      ↓
R3
Fine-Tune layer3 + layer4
```

while:

```text
R4
From Scratch
```

acts as an independent reference configuration.

The central question is therefore whether progressively adapting the pretrained representation improves validation performance.

A useful result pattern would be one in which:

$$
R1 < R2 < R3
$$

in terms of the selected validation metrics.

However, such an ordering should not be assumed before observing the experimental results.

If R3 does not outperform R2, this may indicate that deeper fine-tuning does not provide additional benefit under the current dataset and optimization configuration.

Likewise, if R1 performs competitively with R2 or R3, the pretrained features may already contain sufficiently useful representations for this task.

---

# 16. Reproducibility

The experiment configuration is explicitly defined through YAML rather than hard-coded training parameters.

The main reproducibility controls include:

```yaml
seed: 42
```

and explicit definitions for:

* dataset location
* split ratios
* model strategy
* pretrained initialization
* image size
* batch size
* normalization
* optimizer
* learning rate
* weight decay
* scheduler
* epoch count
* checkpoint source

This makes the ResNet phase easier to reproduce and compare across future experiments.

---

# 17. Summary

The ResNet phase evaluates four progressively different approaches to vehicle classification.

R1 establishes a pretrained feature-extraction baseline.

R2 investigates adaptation of the final ResNet block through controlled fine-tuning.

R3 extends this adaptation into a deeper layer while using progressively smaller learning rates for pretrained parameters.

R4 provides an independent from-scratch configuration to investigate the role of pretrained representations.

The overall experimental design can therefore be summarized as:

```text
                 ResNet18
                    │
          ┌─────────┴─────────┐
          │                   │
     ImageNet             No Pretraining
     Pretrained                 │
          │                     │
         R1                    R4
          │
     Fine-tune layer4
          │
         R2
          │
     Fine-tune layer3
          │
         R3
```

The final conclusion of this phase should be based on the measured validation metrics, training behavior, confusion matrices, and error analysis rather than validation accuracy alone.
