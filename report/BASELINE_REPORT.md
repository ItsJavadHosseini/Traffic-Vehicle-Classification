# Baseline Model Experiments

## 1. Overview

The baseline phase was designed to establish a reliable reference point for the traffic vehicle classification task before moving to pretrained architectures such as ResNet.

The objective was not simply to maximize validation accuracy. Instead, the experiments investigated how several fundamental training choices affect:

* Validation accuracy
* Macro Precision
* Macro Recall
* Macro F1
* Validation loss
* Train/validation generalization gap
* Class-level weaknesses

The baseline model was intentionally trained from scratch so that the effect of the dataset, architecture, regularization, pooling strategy, and optimization strategy could be understood before introducing transfer learning.

The experiments covered:

1. No augmentation vs. augmentation
2. Dropout strength
3. Pooling strategy
4. Weight decay
5. Learning-rate scheduling
6. Reproducibility of the training pipeline

---

# 2. Baseline Architecture

All experiments were based on the same `deep_baseline` CNN architecture.

The model uses four convolutional stages with increasing feature capacity:

```text
Input
  │
  ├── Conv Block: 3 → 32
  ├── Conv Block: 32 → 64
  ├── Conv Block: 64 → 128
  ├── Conv Block: 128 → 256
  │
  ├── Global Pooling
  ├── Dropout
  └── Fully Connected Classifier
        │
        └── 8 vehicle classes
```

The convolutional blocks use Batch Normalization and ReLU activation, while spatial downsampling is performed throughout the network.

The baseline was trained with:

```text
Loss:       Cross Entropy
Optimizer:  AdamW
Learning Rate: 0.001
Weight Decay:  0.0 (default baseline)
Input Size:   224 × 224
```

The primary baseline configuration was:

```yaml
model:
  name: deep_baseline
  dropout: 0.3
  pooling: max

loss:
  name: cross_entropy

optimizer:
  name: adamw
  lr: 0.001
  weight_decay: 0.0

scheduler:
  name: none

augmentation:
  enabled: false
```

This configuration is referred to as the **reference baseline**.

---

# 3. Experimental Design

The experiments were conducted incrementally rather than changing multiple major factors simultaneously.

The main reference configuration was:

> `dropout=0.3 + max pooling + AdamW + lr=0.001 + weight_decay=0 + no scheduler + no augmentation`

Individual experiments then changed one major factor at a time where possible.

This makes it possible to attribute performance differences to specific configuration changes instead of treating the model as a black box.

---

# 4. Experiment Results

## 4.1 Overall Results

| Exp    | Configuration       | Best Val Acc |   Macro F1 | Best Val Loss | Generalization Gap |
| ------ | ------------------- | -----------: | ---------: | ------------: | -----------------: |
| **01** | No Augmentation     |   **86.56%** | **86.01%** |        0.4480 |          **5.72%** |
| **02** | Augmentation        |       77.89% |     78.02% |        0.6272 |             26.26% |
| **03** | Dropout = 0.0       |       84.92% |     84.57% |        0.4737 |              8.42% |
| **04** | Dropout = 0.3       |   **86.56%** | **86.01%** |        0.4480 |          **5.72%** |
| **05** | Dropout = 0.5       |       82.54% |     82.46% |        0.4918 |             13.01% |
| **06** | Max Pooling         |   **86.56%** | **86.01%** |        0.4480 |          **5.72%** |
| **07** | Average Pooling     |       85.43% |     85.22% |    **0.4512** |          **4.50%** |
| **08** | Weight Decay = 0    |   **86.56%** | **86.01%** |        0.4480 |          **5.72%** |
| **09** | Weight Decay = 1e-4 |       81.03% |     81.01% |        0.5534 |             21.02% |
| **10** | Constant LR         |   **86.56%** | **86.01%** |        0.4480 |          **5.72%** |
| **11** | StepLR              |       85.18% |     84.87% |    **0.4086** |              6.35% |
| **12** | ReduceLROnPlateau   |   **88.57%** | **88.45%** |    **0.3550** |              8.55% |

### Best baseline result

The strongest configuration was:

```text
Experiment:        Exp12
Scheduler:         ReduceLROnPlateau
Dropout:           0.3
Pooling:           Max
Learning Rate:     0.001
Weight Decay:      0
Augmentation:      Disabled
```

Performance:

```text
Best Validation Accuracy:  88.57%
Macro Precision:           88.84%
Macro Recall:              88.37%
Macro F1:                  88.45%
Best Validation Loss:       0.3550
```

---

# 5. Experiment 01 — No Augmentation

### Configuration

```text
Dropout:       0.3
Pooling:       Max
Augmentation:  None
Optimizer:     AdamW
LR:            0.001
Weight Decay:  0
Scheduler:     None
```

### Results

```text
Best Epoch:            24
Best Val Loss:         0.4480
Best Val Accuracy:     86.56%
Macro Precision:       86.43%
Macro Recall:          86.50%
Macro F1:              86.01%

Final Train Accuracy:  91.27%
Final Val Accuracy:    85.55%

Generalization Gap:    5.72%
```

This experiment established the first meaningful reference point for the project.

The model achieved approximately **86.6% validation accuracy** while maintaining a relatively moderate generalization gap.

The class with the lowest recall was:

```text
kamyunet
```

while the lowest precision was observed for:

```text
ambulance
```

This already indicated that the main difficulty was not necessarily global classification capacity, but distinguishing visually similar vehicle categories.

---

# 6. Experiment 02 — Data Augmentation

The second experiment introduced:

```text
Horizontal Flip
Rotation:      ±15°
Color Jitter
Random Crop
```

### Results

```text
Best Epoch:            26
Best Val Accuracy:     77.89%
Macro F1:              78.02%
Best Val Loss:          0.6272
Generalization Gap:     26.26%
```

This was a significant degradation compared with the baseline.

Validation accuracy dropped from:

```text
86.56% → 77.89%
```

representing an absolute decrease of approximately:

```text
-8.67 percentage points
```

The final model also showed:

```text
Train Accuracy: 87.19%
Validation Accuracy: 60.93%
```

with a very large generalization gap.

### Interpretation

The result suggests that this augmentation policy was not appropriate for the current dataset/model combination.

Possible explanations include:

* Some transformations may not preserve the discriminative visual characteristics of the vehicle classes.
* Random cropping may remove important vehicle features.
* Color jitter may modify visual information that is useful for classification.
* The relatively small dataset may make aggressive augmentation introduce a distribution that differs substantially from the validation data.

The important conclusion is not that augmentation is inherently harmful.

Rather:

> **This particular augmentation policy was harmful for this dataset.**

This distinction is important because augmentation should be treated as a hypothesis to test rather than an automatic improvement.

---

# 7. Experiment 03 — Dropout = 0

The baseline dropout of `0.3` was removed.

### Results

```text
Best Epoch:          29
Best Val Accuracy:   84.92%
Macro F1:            84.57%
Best Val Loss:        0.4737
Generalization Gap:   8.42%
```

Compared with the reference baseline:

```text
86.56% → 84.92% validation accuracy
```

and:

```text
86.01% → 84.57% Macro F1
```

Training accuracy increased to:

```text
93.35%
```

while validation performance decreased.

This is consistent with weaker regularization allowing the network to fit the training data more aggressively without producing better generalization.

---

# 8. Experiment 05 — Dropout = 0.5

Increasing dropout from `0.3` to `0.5` produced:

```text
Best Val Accuracy: 82.54%
Macro F1:          82.46%
Generalization Gap: 13.01%
```

Performance again decreased.

The comparison:

```text
Dropout 0.0 → 84.92%
Dropout 0.3 → 86.56%
Dropout 0.5 → 82.54%
```

shows that, within the tested range, `0.3` provided the strongest validation performance.

Interestingly, increasing dropout did not simply improve generalization.

This demonstrates that regularization has an appropriate range: too little can allow overfitting, while too much can reduce the model's ability to learn useful representations.

---

# 9. Experiments 04, 06, 08 and 10 — Reproducibility Controls

Experiments 04, 06, 08 and 10 produced exactly the same configuration/results as Experiment 01.

Their results were:

```text
Best Val Accuracy: 86.56%
Macro F1:          86.01%
Best Val Loss:      0.4480
Generalization Gap: 5.72%
```

Although these experiments do not represent new hyperparameter conditions, their consistency is useful.

Repeated identical results indicate that the current experimental pipeline is behaving reproducibly under the tested conditions.

For the final report, these runs should therefore be treated as **control/replication runs rather than separate model improvements**.

---

# 10. Experiment 07 — Average Pooling

The pooling strategy was changed from max pooling to average pooling.

### Results

```text
Best Val Accuracy: 85.43%
Macro F1:          85.22%
Best Val Loss:      0.4512
Generalization Gap: 4.50%
```

Compared with max pooling:

```text
Max Pooling:
Accuracy = 86.56%
F1       = 86.01%

Average Pooling:
Accuracy = 85.43%
F1       = 85.22%
```

Max pooling therefore produced slightly higher classification performance in this experiment.

However, average pooling produced the smallest generalization gap:

```text
4.50%
```

This suggests that average pooling produced somewhat more conservative generalization behavior, although not enough to compensate for the reduction in validation accuracy and F1.

---

# 11. Experiment 09 — Weight Decay

Weight decay was introduced:

```text
Weight Decay = 1e-4
```

### Results

```text
Best Val Accuracy: 81.03%
Macro F1:          81.01%
Best Val Loss:      0.5534
Generalization Gap: 21.02%
```

This was substantially worse than the reference configuration.

The model reached:

```text
Train Accuracy: 91.87%
Validation Accuracy: 70.85% (final)
```

indicating that the tested weight decay value did not provide the desired regularization effect.

This does not imply that weight decay should never be used. It only shows that `1e-4`, under the current optimizer, learning rate, dataset and architecture, was not beneficial.

---

# 12. Experiment 11 — StepLR

A StepLR scheduler was introduced:

```text
step_size = 5
gamma = 0.5
```

### Results

```text
Best Epoch:          28
Best Val Loss:        0.4086
Best Val Accuracy:   85.18%
Macro F1:            84.87%
Final Val Accuracy:  86.18%
Generalization Gap:   6.35%
```

An interesting result appeared here.

The best validation loss improved considerably:

```text
0.4480 → 0.4086
```

but the best validation accuracy did not improve over the baseline:

```text
86.56% → 85.18%
```

This demonstrates why relying on a single metric can be misleading.

Lower validation loss does not necessarily translate directly into higher classification accuracy.

The scheduler was therefore useful from an optimization perspective, but it did not produce the strongest classification result among the tested configurations.

---

# 13. Experiment 12 — ReduceLROnPlateau

The final baseline experiment introduced:

```text
Scheduler: ReduceLROnPlateau
Patience: 2
Factor: 0.5
```

All other major parameters remained equal to the reference baseline.

### Results

```text
Best Epoch:            28

Best Val Loss:          0.3550
Best Val Accuracy:     88.57%

Macro Precision:       88.84%
Macro Recall:          88.37%
Macro F1:              88.45%

Final Train Accuracy:  95.35%
Final Val Accuracy:    86.81%

Generalization Gap:     8.55%
```

This was the strongest experiment in the baseline phase.

Compared with the reference baseline:

```text
Validation Accuracy:
86.56% → 88.57%
+2.01 percentage points

Macro F1:
86.01% → 88.45%
+2.44 percentage points

Validation Loss:
0.4480 → 0.3550
-0.0930
```

The improvement is particularly meaningful because it was achieved without increasing the model architecture or adding external pretrained knowledge.

The main change was the learning-rate adaptation strategy.

---

# 14. Cross-Experiment Analysis

## 14.1 Regularization

The dropout experiments produced the following pattern:

```text
Dropout 0.0   → 84.92%
Dropout 0.3   → 86.56%
Dropout 0.5   → 82.54%
```

The tested results suggest that `0.3` provided a better balance between fitting the training data and maintaining validation performance.

The experiment with no dropout reached a higher training accuracy but did not generalize as well.

Increasing dropout to `0.5`, however, reduced performance rather than improving it.

Therefore:

> `Dropout = 0.3` was retained as the baseline regularization level.

---

## 14.2 Pooling

The comparison was:

```text
Max Pooling  → 86.56% accuracy / 86.01% F1
Avg Pooling  → 85.43% accuracy / 85.22% F1
```

Max pooling produced stronger classification metrics in the current task.

Average pooling had a slightly smaller generalization gap, but the difference was not sufficient to compensate for the lower validation performance.

Therefore:

> Max pooling was retained.

---

## 14.3 Data Augmentation

The augmentation experiment produced the largest negative change:

```text
No Augmentation → 86.56%
Augmentation    → 77.89%
```

The resulting generalization gap also increased dramatically.

This indicates that the selected augmentation pipeline was not aligned with the current dataset.

This result is especially important because it prevents blindly assuming that "more augmentation = better generalization."

Future augmentation experiments should be more targeted and dataset-driven.

For example, individual transformations could be evaluated separately rather than introducing:

```text
flip + rotation + color jitter + random crop
```

as a single package.

---

## 14.4 Weight Decay

The tested weight decay:

```text
0 → 1e-4
```

produced:

```text
86.56% → 81.03%
```

Therefore the tested value was not retained.

A larger experimental search could investigate whether the problem was specifically the magnitude of `1e-4`, but there was no evidence in the current experiments that this setting improved the model.

---

## 14.5 Learning-Rate Scheduling

This was the most important optimization result.

The reference configuration used a constant learning rate:

```text
lr = 0.001
```

StepLR improved validation loss but not the primary classification metrics.

ReduceLROnPlateau produced the strongest overall result:

```text
88.57% validation accuracy
88.45% Macro F1
0.355 validation loss
```

The model therefore benefited from adapting the learning rate based on validation behavior.

This suggests that the optimization landscape of the baseline model was not fully exploited by maintaining a constant learning rate throughout training.

---

# 15. Class-Level Observations

A recurring weakness across several experiments was:

```text
kamyunet
```

which repeatedly appeared among the lowest-recall classes.

This is consistent with the nature of the dataset: several vehicle categories have strong visual similarity.

The difficulty therefore appears to be concentrated around specific class boundaries rather than being uniformly distributed across all eight classes.

Other low-performing classes varied depending on the experiment:

```text
ambulance
kamyun
autobus
vanet
```

This variation suggests that changing optimization and regularization affects the decision boundaries differently across classes.

A future error-analysis phase should therefore inspect the confusion matrix and representative false predictions, especially among visually similar categories.

---

# 16. Generalization Analysis

The train/validation relationship is particularly informative.

For example, Experiment 12 achieved:

```text
Train Accuracy: 95.35%
Validation Accuracy: 86.81%
```

at the end of training.

The corresponding gap was:

```text
8.55 percentage points
```

This indicates that the model still has some degree of overfitting, but the validation performance remained strong.

By contrast, Experiment 02 reached a much more problematic state:

```text
Train Accuracy: 87.19%
Final Val Accuracy: 60.93%
Gap: 26.26 percentage points
```

Therefore, the augmentation experiment did not simply make the model "more robust." Under this configuration, it produced a substantially larger train/validation discrepancy.

This reinforces the importance of evaluating augmentation empirically.

---

# 17. Key Findings

The baseline experiments produced several clear conclusions.

### Finding 1 — The CNN baseline is already a meaningful classifier

Without transfer learning, the model reached:

```text
88.57% Best Validation Accuracy
88.45% Macro F1
```

This establishes a strong reference point for evaluating pretrained architectures.

---

### Finding 2 — ReduceLROnPlateau produced the strongest baseline

Among the tested configurations, the adaptive learning-rate schedule produced the best validation accuracy, macro precision, macro recall and macro F1.

```text
Best Configuration:

Deep Baseline
+ Dropout 0.3
+ Max Pooling
+ AdamW
+ LR 0.001
+ Weight Decay 0
+ ReduceLROnPlateau
+ No Augmentation
```

---

### Finding 3 — Aggressive augmentation was counterproductive

The tested augmentation policy reduced validation accuracy by approximately:

```text
8.67 percentage points
```

and produced a much larger generalization gap.

The correct conclusion is not to remove augmentation permanently, but to redesign it based on the actual image distribution.

---

### Finding 4 — Moderate dropout performed better than both extremes

The tested results favored:

```text
Dropout = 0.3
```

over:

```text
0.0
0.5
```

---

### Finding 5 — Max pooling slightly outperformed average pooling

Max pooling achieved higher validation accuracy and macro F1.

Average pooling did produce a smaller generalization gap, but not higher classification performance.

---

### Finding 6 — Optimization strategy mattered more than several architecture-level changes

The baseline architecture itself was kept constant across experiments.

The most meaningful improvement came from changing how the learning rate was controlled rather than making the CNN deeper or wider.

This is an important baseline finding because it shows that optimization choices can materially affect performance even when model capacity remains unchanged.

---

# 18. Baseline Configuration Selected for the Next Phase

Based on the experimental results, the baseline configuration selected as the reference model for the next stage is:

```yaml
model:
  name: deep_baseline
  dropout: 0.3
  pooling: max

loss:
  name: cross_entropy

optimizer:
  name: adamw
  lr: 0.001
  weight_decay: 0.0

scheduler:
  name: reduce_on_plateau
  patience: 2
  factor: 0.5

augmentation:
  enabled: false
```

Performance reference:

```text
Validation Accuracy: 88.57%
Macro Precision:     88.84%
Macro Recall:        88.37%
Macro F1:            88.45%
Validation Loss:      0.3550
```

---

# 19. Limitations

Several limitations should be considered when interpreting these experiments.

First, the experiments were performed on a relatively small dataset of approximately 3.6K training images across eight classes. Therefore, small changes in the train/validation split or image distribution may affect the reported metrics.

Second, the augmentation experiment tested several transformations simultaneously. Consequently, the individual contribution of horizontal flipping, rotation, color jitter and random cropping cannot be isolated from this experiment.

Third, only one non-zero weight-decay value was tested. The poor result of `1e-4` should therefore not be interpreted as evidence that weight decay itself is universally ineffective.

Fourth, the current analysis focuses primarily on aggregate metrics. Class-level confusion matrices and visual error analysis are required to understand why visually similar classes such as `kamyun`, `kamyunet`, and related categories remain difficult.

Finally, the baseline was trained from scratch. It therefore does not directly represent the upper bound of what can be achieved using transfer learning.

---

# 20. Conclusion

The baseline phase established a reproducible CNN reference model and evaluated several fundamental training choices.

The experiments demonstrated that the strongest configuration was not obtained by simply increasing model regularization or adding aggressive augmentation. Instead, the largest useful improvement came from adapting the learning rate during training.

The final baseline achieved:

```text
88.57% Validation Accuracy
88.45% Macro F1
0.355 Validation Loss
```

using a relatively simple CNN trained from scratch.

This provides a clear benchmark for the next phase of the project.

The next architectural question is therefore not:

> "Can we make the baseline CNN more complicated?"

but rather:

> "How much additional representation quality can a pretrained architecture provide over this 88.57% baseline?"

This establishes the baseline as a quantitative reference for the subsequent ResNet experiments.

The transition to a pretrained model should therefore preserve the experimental discipline established here: controlled changes, comparable metrics, class-level analysis, and explicit measurement of the generalization gap.

## Baseline Experiment Results

| Exp | Experiment | Val Acc ↑ | Macro F1 ↑ | Best Val Loss ↓ | Gap |
|:---:|:---|---:|---:|---:|---:|
| 01 | No Augmentation | **86.56%** | **86.01%** | 0.4480 | 5.72% |
| 02 | Augmentation | 77.89% | 78.02% | 0.6272 | 26.26% |
| 03 | Dropout = 0.0 | 84.92% | 84.57% | 0.4737 | 8.42% |
| 04 | Dropout = 0.3 (Control) | **86.56%** | **86.01%** | 0.4480 | 5.72% |
| 05 | Dropout = 0.5 | 82.54% | 82.46% | 0.4918 | 13.01% |
| 06 | Max Pooling (Control) | **86.56%** | **86.01%** | 0.4480 | 5.72% |
| 07 | Average Pooling | 85.43% | 85.22% | **0.4512** | **4.50%** |
| 08 | Weight Decay = 0 (Control) | **86.56%** | **86.01%** | 0.4480 | 5.72% |
| 09 | Weight Decay = 1e-4 | 81.03% | 81.01% | 0.5534 | 21.02% |
| 10 | Constant LR (Control) | **86.56%** | **86.01%** | 0.4480 | 5.72% |
| 11 | StepLR | 85.18% | 84.87% | **0.4086** | 6.35% |
| 12 | ReduceLROnPlateau | **88.57%** | **88.45%** | **0.3550** | 8.55% |