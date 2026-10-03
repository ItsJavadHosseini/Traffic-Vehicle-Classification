from pathlib import Path

import torch
from PIL import Image

from prediction.preprocessing import get_resnet_eval_transform
from src.models.m_resnet import ResNet18Model


class ResNetPredictor:

    def __init__(
        self,
        checkpoint_path,
        image_size=224,
        device=None,
    ):
        self.checkpoint_path = Path(
            checkpoint_path
        )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        # ----------------------------------------------------
        # Class names
        # ----------------------------------------------------

        self.class_names = self.checkpoint[
            "class_names"
        ]

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.model = ResNet18Model(
            num_classes=len(self.class_names),
            strategy="fine_tune_stage_2",
            pretrained=False,
            small_input=False,
        )

        self.model.load_state_dict(
            self.checkpoint[
                "model_state_dict"
            ]
        )

        self.model.to(self.device)

        self.model.eval()

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        self.transform = (
            get_resnet_eval_transform(
                image_size=image_size
            )
        )

    # ========================================================
    # Prediction
    # ========================================================

    @torch.no_grad()
    def predict(self, image):

        # ----------------------------------------------------
        # PIL image
        # ----------------------------------------------------

        if not isinstance(image, Image.Image):
            raise TypeError(
                "image must be a PIL.Image.Image"
            )

        image = image.convert("RGB")

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        image_tensor = self.transform(
            image
        )

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(self.device)
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        outputs = self.model(
            image_tensor
        )

        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predicted_index = (
            probabilities
            .argmax(dim=1)
            .item()
        )

        predicted_class = (
            self.class_names[
                predicted_index
            ]
        )

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