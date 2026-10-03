# Traffic Vehicle Classification

## Baseline Model Experiments Report

### 1. Overview

The baseline phase was designed to establish a reliable reference model for the traffic vehicle classification task before evaluating pretrained architectures such as ResNet18.

The purpose of this phase was not simply to maximize validation accuracy. Instead, the experiments investigated how fundamental modeling and training choices affect classification performance, optimization behavior, and generalization.

The evaluated factors included:

* Data augmentation
* Dropout strength
* Pooling strategy
* Weight decay
* Learning-rate scheduling
* Reproducibility of the training pipeline

The baseline model was trained from scratch. This provided a controlled reference point for evaluating the additional representation quality provided by pretrained architectures in the subsequent ResNet experiments.

---

## 2. Baseline Architecture

All baseline experiments used the same `deep_baseline` CNN architecture. The experiments therefore focused primarily on training and regularization choices rather than changing the underlying network capacity.

The architecture consists of four convolutional stages with increasing feature capacity:

```text
Input Image
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
            └── 8 Vehicle Classes
```

The convolutional blocks use Batch Normalization and ReLU activation, with spatial downsampling throughout the network.

The reference configuration was:

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

The reference training setup therefore used:

```text
Loss:          Cross Entropy
Optimizer:     AdamW
Learning Rate: 0.001
Weight Decay:  0
Input Size:    224 × 224
Dropout:       0.3
Pooling:       Max Pooling
Augmentation:  Disabled
Scheduler:     None
```

---

## 3. Experimental Design

The experiments were conducted incrementally, with individual experiments changing one major factor where possible.

The reference configuration was:

```text
Dropout = 0.3
Max Pooling
AdamW
Learning Rate = 0.001
Weight Decay = 0
No Scheduler
No Augmentation
```

The experiments then evaluated individual alternatives.

This design provides a more interpretable comparison because performance changes can be associated with specific configuration changes rather than treating the model as a black box.

Some experiments were repeated with the reference configuration as control/replication runs. These runs are useful for assessing the reproducibility of the training pipeline but should not be interpreted as independent hyperparameter configurations.

---

# 4. Overall Results

| Exp | Configuration              | Best Val Acc |   Macro F1 | Best Val Loss | Train–Val Gap |
| --: | -------------------------- | -----------: | ---------: | ------------: | ------------: |
|  01 | No Augmentation            |   **86.56%** | **86.01%** |        0.4480 |     **5.72%** |
|  02 | Augmentation               |       77.89% |     78.02% |        0.6272 |        26.26% |
|  03 | Dropout = 0.0              |       84.92% |     84.57% |        0.4737 |         8.42% |
|  04 | Dropout = 0.3 — Control    |   **86.56%** | **86.01%** |        0.4480 |     **5.72%** |
|  05 | Dropout = 0.5              |       82.54% |     82.46% |        0.4918 |        13.01% |
|  06 | Max Pooling — Control      |   **86.56%** | **86.01%** |        0.4480 |     **5.72%** |
|  07 | Average Pooling            |       85.43% |     85.22% |        0.4512 |     **4.50%** |
|  08 | Weight Decay = 0 — Control |   **86.56%** | **86.01%** |        0.4480 |     **5.72%** |
|  09 | Weight Decay = 1e-4        |       81.03% |     81.01% |        0.5534 |        21.02% |
|  10 | Constant LR — Control      |   **86.56%** | **86.01%** |        0.4480 |     **5.72%** |
|  11 | StepLR                     |       85.18% |     84.87% |    **0.4086** |         6.35% |
|  12 | ReduceLROnPlateau          |   **88.57%** | **88.45%** |    **0.3550** |         8.55% |

The strongest validation result was obtained by Experiment 12 using `ReduceLROnPlateau`.

---

# 5. Reference Baseline — Experiment 01

The first experiment established the initial reference point for the project.

Configuration:

```text
Dropout:       0.3
Pooling:       Max
Augmentation:  None
Optimizer:     AdamW
Learning Rate: 0.001
Weight Decay:  0
Scheduler:     None
```

Results:

```text
Best Epoch:            24
Best Validation Loss:   0.4480
Best Validation Acc:   86.56%

Macro Precision:       86.43%
Macro Recall:          86.50%
Macro F1:              86.01%

Final Train Accuracy:  91.27%
Final Validation Acc:  85.55%

Train–Val Gap:          5.72 percentage points
```

The model established a validation accuracy of approximately 86.6%.

At the class level, `kamyunet` was among the weakest classes by recall, while `ambulance` showed the lowest precision in this experiment.

This suggested that classification difficulty was concentrated around particular class boundaries rather than being uniformly distributed across all eight categories.

---

# 6. Experiment 02 — Data Augmentation

The second experiment introduced the following augmentation policy:

```text
Horizontal Flip
Rotation: ±15°
Color Jitter
Random Crop
```

Results:

```text
Best Epoch:             26
Best Validation Acc:    77.89%
Macro F1:               78.02%
Best Validation Loss:    0.6272
Train–Val Gap:          26.26 percentage points
```

Validation accuracy decreased from:

```text
86.56% → 77.89%
```

This represents an absolute decrease of approximately:

```text
8.67 percentage points
```

The final training and validation accuracies were:

```text
Train Accuracy:       87.19%
Validation Accuracy:  60.93%
```

The results indicate that the tested augmentation policy was not beneficial for this dataset and training setup.

This should not be interpreted as evidence that augmentation is inherently harmful. Rather, the specific combination of transformations appears to have introduced a training distribution that did not improve validation performance.

Potential factors include:

* Random cropping removing useful vehicle features.
* Color jitter modifying visually informative characteristics.
* Rotation altering image patterns that may be relevant to the classes.
* The combined augmentation policy being too aggressive for the current dataset.

A more controlled future study could evaluate individual augmentation techniques separately.

---

# 7. Dropout Experiments

## 7.1 Dropout = 0.0

Removing dropout resulted in:

```text
Best Validation Accuracy: 84.92%
Macro F1:                 84.57%
Best Validation Loss:      0.4737
Train–Val Gap:             8.42 percentage points
```

Compared with the reference configuration:

```text
86.56% → 84.92%
```

Training accuracy increased to approximately 93.35%, while validation performance decreased.

This indicates that removing dropout allowed the network to fit the training data more strongly without producing better validation performance.

---

## 7.2 Dropout = 0.5

Increasing dropout to 0.5 produced:

```text
Best Validation Accuracy: 82.54%
Macro F1:                 82.46%
Best Validation Loss:      0.4918
Train–Val Gap:            13.01 percentage points
```

The three tested dropout configurations produced:

```text
Dropout 0.0 → 84.92%
Dropout 0.3 → 86.56%
Dropout 0.5 → 82.54%
```

Within the tested range, `dropout=0.3` produced the strongest validation performance.

The results also demonstrate that stronger regularization does not automatically produce better generalization. Excessive dropout can reduce the model's ability to learn useful representations.

---

# 8. Pooling Experiment

The pooling strategy was changed from max pooling to average pooling.

### Max Pooling

```text
Validation Accuracy: 86.56%
Macro F1:             86.01%
Validation Loss:       0.4480
Train–Val Gap:          5.72%
```

### Average Pooling

```text
Validation Accuracy: 85.43%
Macro F1:             85.22%
Validation Loss:       0.4512
Train–Val Gap:          4.50%
```

Average pooling produced a slightly smaller train–validation accuracy gap, but max pooling produced higher validation accuracy and Macro F1.

Therefore, max pooling was retained for the reference configuration.

---

# 9. Weight Decay Experiment

Weight decay was introduced with:

```text
Weight Decay = 1e-4
```

Results:

```text
Best Validation Accuracy: 81.03%
Macro F1:                 81.01%
Best Validation Loss:      0.5534
Train–Val Gap:            21.02 percentage points
```

Compared with the reference:

```text
Weight Decay = 0     → 86.56%
Weight Decay = 1e-4  → 81.03%
```

The tested value therefore did not improve performance under the current configuration.

This result should not be interpreted as evidence that weight decay is universally ineffective. Only one non-zero value was evaluated.

---

# 10. Learning-Rate Scheduling

Two scheduling strategies were evaluated against the constant-learning-rate reference.

## 10.1 StepLR

The StepLR configuration used:

```text
Step Size = 5
Gamma = 0.5
```

Results:

```text
Best Validation Accuracy: 85.18%
Macro F1:                 84.87%
Best Validation Loss:      0.4086
Train–Val Gap:              6.35%
```

An important observation was that validation loss improved:

```text
0.4480 → 0.4086
```

while best validation accuracy decreased:

```text
86.56% → 85.18%
```

This demonstrates that different evaluation metrics can move in different directions.

A lower validation loss did not translate into higher classification accuracy in this experiment.

---

# 11. Experiment 12 — ReduceLROnPlateau

The final baseline experiment introduced:

```text
Scheduler: ReduceLROnPlateau
Patience: 2
Factor: 0.5
```

The other major baseline settings remained unchanged.

Results:

```text
Best Epoch:             28

Best Validation Loss:    0.3550
Best Validation Accuracy: 88.57%

Macro Precision:         88.84%
Macro Recall:            88.37%
Macro F1:                88.45%

Final Train Accuracy:    95.35%
Final Validation Acc:    86.81%

Train–Val Gap:            8.55 percentage points
```

This was the strongest validation result among the tested configurations.

Compared with the reference baseline:

```text
                         Reference      Exp12
Validation Accuracy      86.56%         88.57%
Macro F1                 86.01%         88.45%
Validation Loss           0.4480          0.3550
```

The absolute improvement was:

```text
Validation Accuracy: +2.01 percentage points
Macro F1:             +2.44 percentage points
Validation Loss:      -0.0930
```

The improvement was achieved without increasing the architecture's capacity or introducing pretrained representations.

The main experimental change was the learning-rate adaptation strategy.

---

# 12. Reproducibility and Control Runs

Experiments 04, 06, 08 and 10 produced the same configuration and results as the reference experiment.

Their results were:

```text
Validation Accuracy: 86.56%
Macro F1:             86.01%
Validation Loss:       0.4480
Train–Val Gap:          5.72%
```

These runs are best interpreted as control or replication runs rather than independent hyperparameter experiments.

Their consistency provides evidence that the current training pipeline behaves deterministically under the tested conditions.

---

# 13. Cross-Experiment Analysis

## 13.1 Regularization

The dropout experiments showed:

```text
0.0 → 84.92%
0.3 → 86.56%
0.5 → 82.54%
```

The tested results therefore support `dropout=0.3` as the most effective of the three tested values.

Both removing dropout and increasing it substantially reduced validation performance.

---

## 13.2 Pooling

The pooling comparison showed:

```text
Max Pooling → 86.56% accuracy / 86.01% F1
Avg Pooling → 85.43% accuracy / 85.22% F1
```

Max pooling produced stronger classification performance.

Average pooling produced a smaller train–validation gap, but this did not translate into better validation accuracy or Macro F1.

---

## 13.3 Data Augmentation

The tested augmentation policy produced the largest negative change:

```text
No Augmentation → 86.56%
Augmentation     → 77.89%
```

The train–validation gap also increased substantially.

Therefore, the tested augmentation policy was not retained for the selected baseline.

Future augmentation experiments should be more targeted and should ideally evaluate individual transformations or smaller combinations.

---

## 13.4 Weight Decay

The tested value of `1e-4` reduced validation performance substantially.

However, because only one non-zero value was tested, the result should be interpreted as:

> `weight_decay=1e-4` was not beneficial under the current experimental configuration.

It should not be generalized to all possible weight-decay values or optimization setups.

---

## 13.5 Learning-Rate Scheduling

Learning-rate scheduling produced the most meaningful optimization improvement.

StepLR improved validation loss but did not improve the primary classification metrics.

ReduceLROnPlateau produced the strongest combination of:

```text
Validation Accuracy
Macro Precision
Macro Recall
Macro F1
Validation Loss
```

among the tested baseline configurations.

This indicates that adaptive learning-rate reduction based on validation behavior was particularly effective for this model.

---

# 14. Class-Level Analysis

The class-level results indicate that errors are not uniformly distributed across all vehicle categories.

`kamyunet` repeatedly appeared among the weaker classes, particularly in terms of recall.

Other difficult classes varied between experiments, including:

```text
ambulance
kamyun
autobus
vanet
```

This pattern suggests that part of the classification challenge comes from visually similar vehicle categories.

Aggregate accuracy alone cannot explain these errors.

A subsequent error-analysis stage should therefore use:

* Confusion matrix
* Per-class precision
* Per-class recall
* Per-class F1
* Representative false predictions

This will help identify which class boundaries are responsible for the majority of remaining errors.

---

# 15. Generalization Analysis

The relationship between training and validation performance provides additional information beyond the best validation score.

For Experiment 12:

```text
Final Train Accuracy:  95.35%
Final Val Accuracy:    86.81%
```

resulting in an accuracy difference of approximately:

```text
8.55 percentage points
```

This indicates that some degree of overfitting remains.

Experiment 02 showed a substantially larger discrepancy:

```text
Train Accuracy:        87.19%
Final Val Accuracy:    60.93%
Gap:                   26.26 percentage points
```

Therefore, the tested augmentation policy did not improve generalization under this configuration.

The selected baseline still shows a non-zero train–validation gap, so the 88.57% result should be interpreted as a strong validation benchmark rather than evidence of perfect generalization.

---

# 16. Selected Baseline Configuration

Based on the experiments, the following configuration was selected as the reference baseline for the next phase:

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

Reference performance:

```text
Best Validation Accuracy: 88.57%
Macro Precision:           88.84%
Macro Recall:              88.37%
Macro F1:                  88.45%
Best Validation Loss:       0.3550
```

---

# 17. Limitations

Several limitations should be considered when interpreting the results.

First, the dataset contains approximately 3.6K training images across eight classes. Consequently, the reported validation metrics may be sensitive to the train–validation split.

Second, the augmentation experiment changed several transformations simultaneously. The individual effect of flipping, rotation, color jitter, and random cropping therefore cannot be isolated.

Third, only one non-zero weight-decay value was evaluated. The result for `1e-4` does not establish that weight decay itself is ineffective.

Fourth, the current evaluation is based on validation data rather than an independent test set. Therefore, the reported 88.57% should be considered a validation benchmark.

Finally, aggregate metrics do not fully explain the remaining classification errors. Class-level confusion analysis and visual inspection of false predictions are required for a deeper understanding of the model's weaknesses.

---

# 18. Baseline-to-ResNet Experimental Protocol

The selected baseline establishes the quantitative reference for the next architectural phase.

The primary comparison metrics should remain consistent:

```text
Best Validation Accuracy
Macro Precision
Macro Recall
Macro F1
Best Validation Loss
Train–Validation Accuracy Gap
```

Where possible, the same dataset split and evaluation procedure should be maintained when evaluating ResNet architectures.

The main question of the next phase is therefore:

> How much additional representation quality can a pretrained ResNet provide compared with the 88.57% validation-accuracy baseline?

The ResNet experiments should be interpreted relative to this reference rather than as isolated model results.

---

# 19. Conclusion

The baseline phase established a reproducible CNN reference model and evaluated several fundamental modeling and optimization choices.

The strongest configuration was obtained without increasing the model architecture or using pretrained knowledge.

The selected model achieved:

```text
88.57% Best Validation Accuracy
88.45% Macro F1
88.84% Macro Precision
88.37% Macro Recall
0.3550 Best Validation Loss
```

The most significant improvement came from replacing the constant learning rate with `ReduceLROnPlateau`.

The experiments also demonstrated several important negative results:

* The tested augmentation policy reduced validation performance.
* Removing dropout reduced validation performance.
* Increasing dropout to 0.5 also reduced performance.
* Average pooling slightly underperformed max pooling.
* The tested weight decay of `1e-4` was not beneficial.
* StepLR improved validation loss but did not improve classification accuracy.

These results provide a clear empirical foundation for the next stage of the project.

The baseline therefore serves as a quantitative benchmark against which the ResNet experiments can be evaluated.

The next phase can now investigate whether transfer learning and controlled fine-tuning can improve upon the:

```text
88.57% Validation Accuracy
88.45% Macro F1
```

baseline while maintaining acceptable generalization and computational cost.
