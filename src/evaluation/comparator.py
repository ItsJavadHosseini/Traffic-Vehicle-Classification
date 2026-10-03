from dataclasses import dataclass

import pandas as pd

from .evaluator import EvaluationResult


@dataclass
class ExperimentResult:
    experiment_name: str
    model_name: str
    metrics: EvaluationResult
    best_epoch: int | None = None
    checkpoint_path: str | None = None


class ModelComparator:

    def __init__(self, primary_metric: str = "macro_f1"):
        self.primary_metric = primary_metric
        self.results: list[ExperimentResult] = []

    def add_result(self, result: ExperimentResult) -> None:
        self.results.append(result)

    def compare(self) -> pd.DataFrame:

        if not self.results:
            raise ValueError("No experiment results have been added.")

        rows = []

        for result in self.results:

            row = {
                "experiment_name": result.experiment_name,
                "model_name": result.model_name,
                "best_epoch": result.best_epoch,
                "checkpoint_path": result.checkpoint_path,
                "accuracy": result.metrics.accuracy,
                "macro_f1": result.metrics.macro_f1,
                "macro_precision": result.metrics.macro_precision,
                "macro_recall": result.metrics.macro_recall,
                "weighted_f1": result.metrics.weighted_f1,
                "roc_auc": result.metrics.roc_auc,
            }

            rows.append(row)

        df = pd.DataFrame(rows)

        return df.sort_values(
            by=self.primary_metric,
            ascending=False,
        ).reset_index(drop=True)