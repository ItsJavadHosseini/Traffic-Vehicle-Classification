from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    weighted_f1: float
    roc_auc: float | None


class Evaluator:

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> EvaluationResult:

        accuracy = accuracy_score(y_true, y_pred)

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        macro_precision = precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        macro_recall = recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        roc_auc = None

        if y_prob is not None:
            roc_auc = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
            )

        return EvaluationResult(
            accuracy=accuracy,
            macro_f1=macro_f1,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            weighted_f1=weighted_f1,
            roc_auc=roc_auc,
        )

    @torch.no_grad()
    def predict(
        self,
        model: torch.nn.Module,
        dataloader,
        device: torch.device,
    ):
        model.eval()

        all_targets = []
        all_predictions = []
        all_probabilities = []

        for images, targets in dataloader:

            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1)

            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_predictions)
        y_prob = np.concatenate(all_probabilities)

        return y_true, y_pred, y_prob

    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader,
        device: torch.device,
    ) -> EvaluationResult:

        y_true, y_pred, y_prob = self.predict(
            model=model,
            dataloader=dataloader,
            device=device,
        )

        return self.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )