import numpy as np

from evaluation.evaluator import Evaluator
from evaluation.comparator import (
    ExperimentResult,
    ModelComparator,
)


def main():
    evaluator = Evaluator()

    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])

    y_prob = np.array([
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
        [0.80, 0.10, 0.10],
        [0.10, 0.80, 0.10],
        [0.10, 0.70, 0.20],
    ])

    metrics = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    print("Evaluation Result:")
    print(metrics)

    result = ExperimentResult(
        experiment_name="test_exp",
        model_name="TestModel",
        metrics=metrics,
        best_epoch=10,
        checkpoint_path="test/best.pt",
    )

    comparator = ModelComparator(
        primary_metric="macro_f1"
    )

    comparator.add_result(result)

    print("\nComparison:")
    print(comparator.compare())


if __name__ == "__main__":
    main()