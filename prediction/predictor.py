from pathlib import Path

import torch
from PIL import Image

from prediction.model_registry import MODEL_REGISTRY
from prediction.preprocessing import get_eval_transform
from src.models.m_baseline import DeepBaselineCNN
from src.models.m_resnet import ResNet18Model, ResNet34Model


class ModelPredictor:

    def __init__(
        self,
        model_name,
        device=None,
    ):
        if model_name not in MODEL_REGISTRY:
            available_models = list(MODEL_REGISTRY.keys())

            raise ValueError(
                f"Unknown model: {model_name}\n"
                f"Available models: {available_models}"
            )

        self.model_name = model_name
        self.model_config = MODEL_REGISTRY[model_name]

        self.checkpoint_path = Path(
            self.model_config["checkpoint"]
        )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found:\n"
                f"{self.checkpoint_path}"
            )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        self.class_names = self.checkpoint.get(
            "class_names",
            [
                "ambulance",
                "autobus",
                "kamyun",
                "kamyunet",
                "minibus",
                "savari",
                "taxi",
                "vanet",
            ],
        )

        self.model = self._build_model()

        self.model.load_state_dict(
            self.checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = self._build_transform()

    def _build_model(self):
        architecture = self.model_config["architecture"]

        if architecture == "resnet18":
            strategy = self.checkpoint["strategy"]

            return ResNet18Model(
                num_classes=len(self.class_names),
                strategy=strategy,
                pretrained=False,
                small_input=False,
            )

        if architecture == "resnet34":
            strategy = self.checkpoint["strategy"]

            return ResNet34Model(
                num_classes=len(self.class_names),
                strategy=strategy,
                pretrained=False,
                small_input=False,
            )

        if architecture == "deep_baseline":
            return DeepBaselineCNN(
                num_classes=len(self.class_names),
                dropout=0.3,
                pooling="max",
            )

        raise ValueError(
            f"Unsupported architecture: {architecture}"
        )

    def _build_transform(self):
        architecture = self.model_config["architecture"]

        if architecture in {
            "resnet18",
            "resnet34",
        }:
            experiment_config = self.checkpoint.get(
                "experiment_config",
                {},
            )

            image_size = (
                experiment_config
                .get("data", {})
                .get("image_size", 224)
            )

            return get_eval_transform(
                image_size=image_size
            )

        if architecture == "deep_baseline":
            return get_eval_transform(
                image_size=224
            )

        raise ValueError(
            f"Unsupported architecture: {architecture}"
        )

    @torch.no_grad()
    def predict(self, image):

        if not isinstance(image, Image.Image):
            raise TypeError(
                "image must be a PIL.Image.Image"
            )

        image = image.convert("RGB")

        image_tensor = self.transform(image)

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(self.device)
        )

        outputs = self.model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predicted_index = (
            probabilities
            .argmax(dim=1)
            .item()
        )

        predicted_class = self.class_names[
            predicted_index
        ]

        confidence = (
            probabilities[
                0,
                predicted_index,
            ].item()
        )

        return {
            "class_name": predicted_class,
            "class_index": predicted_index,
            "confidence": confidence,
            "probabilities": (
                probabilities[0]
                .cpu()
                .tolist()
            ),
        }