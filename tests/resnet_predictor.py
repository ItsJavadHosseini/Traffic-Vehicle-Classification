import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path

from PIL import Image

from prediction.predictor import (
    ResNetPredictor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    PROJECT_ROOT
    / "ARTIFACTS"
    / "resnet"
    / "R3"
    / "best_model.pt"
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "ambulance"
    /"195939523.jpg"

)


def main():

    predictor = ResNetPredictor(
        checkpoint_path=CHECKPOINT,
    )

    print("Device:", predictor.device)
    print("Classes:", predictor.class_names)

    image = Image.open(
        IMAGE_PATH
    )

    result = predictor.predict(
        image
    )

    print("\nPrediction")
    print("=" * 50)

    print(
        "Class:",
        result["class_name"]
    )

    print(
        "Index:",
        result["class_index"]
    )

    print(
        "Confidence:",
        f"{result['confidence']:.4f}"
    )

    print("\nProbabilities:")

    for class_name, probability in zip(
        predictor.class_names,
        result["probabilities"],
    ):
        print(
            f"{class_name:12s}: "
            f"{probability:.4f}"
        )


if __name__ == "__main__":
    main()