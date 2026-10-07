import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTModel(nn.Module):
    """
    Vision Transformer (ViT-B/16) for traffic vehicle classification.

    Strategies:
        - feature_extraction: only classification head is trainable
        - fine_tune_stage_1: last encoder blocks + classification head
        - fine_tune_stage_2: more encoder blocks + classification head
        - from_scratch: entire model is trainable
    """

    def __init__(
        self,
        num_classes: int = 8,
        strategy: str = "feature_extraction",
        pretrained: bool = True,
        num_unfrozen_blocks: int = 2,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.strategy = strategy
        self.pretrained = pretrained

        # ---------------------------------------------------------
        # Load ViT-B/16
        # ---------------------------------------------------------
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None

        self.model = vit_b_16(weights=weights)

        # ---------------------------------------------------------
        # Replace classification head
        # ViT-B/16 feature dimension = 768
        # ---------------------------------------------------------
        in_features = self.model.heads.head.in_features

        self.model.heads.head = nn.Linear(
            in_features=in_features,
            out_features=num_classes,
        )

        # ---------------------------------------------------------
        # Configure trainable layers
        # ---------------------------------------------------------
        self._configure_trainable_layers(
            strategy=strategy,
            num_unfrozen_blocks=num_unfrozen_blocks,
        )

    def _configure_trainable_layers(
        self,
        strategy: str,
        num_unfrozen_blocks: int,
    ):
        """
        Configure which ViT parameters are trainable.
        """

        # First freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        # ---------------------------------------------------------
        # Feature Extraction
        # Only classification head is trainable
        # ---------------------------------------------------------
        if strategy == "feature_extraction":

            for param in self.model.heads.head.parameters():
                param.requires_grad = True

        # ---------------------------------------------------------
        # Fine-tuning
        # Unfreeze last N transformer encoder blocks + head
        # ---------------------------------------------------------
        elif strategy in {"fine_tune_stage_1", "fine_tune_stage_2"}:

            num_blocks = len(self.model.encoder.layers)

            start_idx = max(
                0,
                num_blocks - num_unfrozen_blocks,
            )

            for block in self.model.encoder.layers[start_idx:]:
                for param in block.parameters():
                    param.requires_grad = True

            for param in self.model.heads.head.parameters():
                param.requires_grad = True

        # ---------------------------------------------------------
        # Train entire model
        # ---------------------------------------------------------
        elif strategy == "from_scratch":

            for param in self.model.parameters():
                param.requires_grad = True

        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Expected one of: "
                f"feature_extraction, "
                f"fine_tune_stage_1, "
                f"fine_tune_stage_2, "
                f"from_scratch"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Input:
            x: [B, 3, 224, 224]

        Output:
            logits: [B, num_classes]
        """

        return self.model(x)

    def get_trainable_parameters(self):
        """
        Return trainable parameters only.
        """

        return [
            param
            for param in self.parameters()
            if param.requires_grad
        ]

    def get_trainable_parameter_count(self) -> int:
        """
        Number of trainable parameters.
        """

        return sum(
            param.numel()
            for param in self.parameters()
            if param.requires_grad
        )

    def get_total_parameter_count(self) -> int:
        """
        Total number of model parameters.
        """

        return sum(
            param.numel()
            for param in self.parameters()
        )

    def get_parameter_groups(
        self,
        head_lr: float = 1e-3,
        encoder_lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        """
        Create optimizer parameter groups.

        Head gets a larger learning rate than the pretrained
        transformer layers.
        """

        encoder_params = []
        head_params = []

        for name, param in self.named_parameters():

            if not param.requires_grad:
                continue

            if "heads.head" in name:
                head_params.append(param)
            else:
                encoder_params.append(param)

        parameter_groups = []

        if encoder_params:
            parameter_groups.append(
                {
                    "params": encoder_params,
                    "lr": encoder_lr,
                    "weight_decay": weight_decay,
                }
            )

        if head_params:
            parameter_groups.append(
                {
                    "params": head_params,
                    "lr": head_lr,
                    "weight_decay": weight_decay,
                }
            )

        return parameter_groups


if __name__ == "__main__":

    print("=" * 60)
    print("ViT-B/16 Model Test")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    for strategy in [
        "feature_extraction",
        "fine_tune_stage_1",
        "fine_tune_stage_2",
        "from_scratch",
    ]:

        print(f"\nStrategy: {strategy}")

        model = ViTModel(
            num_classes=8,
            strategy=strategy,
            pretrained=True,
            num_unfrozen_blocks=2,
        ).to(device)

        print(
            f"Trainable parameters: "
            f"{model.get_trainable_parameter_count():,}"
        )

        print(
            f"Total parameters: "
            f"{model.get_total_parameter_count():,}"
        )

        x = torch.randn(
            2,
            3,
            224,
            224,
            device=device,
        )

        with torch.no_grad():
            output = model(x)

        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {output.shape}")

        assert output.shape == (2, 8)

    print("\nViT test completed successfully.")