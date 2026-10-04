from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS = PROJECT_ROOT / "ARTIFACTS"


MODEL_REGISTRY = {
    "ResNet18 - R3": {
        "checkpoint": (
            ARTIFACTS
            / "resnet"
            / "R3"
            / "best_model.pt"
        ),
        "architecture": "resnet18",
    },

    "ResNet34 - R34_1": {
        "checkpoint": (
            ARTIFACTS
            / "resnet"
            / "R34_1"
            / "best_model.pt"
        ),
        "architecture": "resnet34",
    },

    "ResNet34 - R34_2": {
        "checkpoint": (
            ARTIFACTS
            / "resnet"
            / "R34_2"
            / "best_model.pt"
        ),
        "architecture": "resnet34",
    },

    "Deep Baseline CNN": {
        "checkpoint": (
            ARTIFACTS
            / "baseline"
            / "exp12_reduce_on_plateau"
            / "best_model.pt"
        ),
        "architecture": "deep_baseline",
    },
}