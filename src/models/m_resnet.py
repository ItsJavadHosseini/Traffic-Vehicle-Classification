import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Model(nn.Module):

    def __init__(
        self,
        num_classes=8,
        strategy="feature_extraction",
        pretrained=True,
        small_input=False,
    ):
        super().__init__()

        self.strategy = strategy
        self.pretrained = pretrained
        self.small_input = small_input

        # ----------------------------------------------------
        # Load ResNet18
        # ----------------------------------------------------

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.backbone = resnet18(
            weights=weights
        )

        # ----------------------------------------------------
        # Small input configuration
        #
        # Used for R4 with 64x64 images.
        #
        # Original ResNet:
        # Conv7x7 / stride 2
        # MaxPool
        #
        # Small-input ResNet:
        # Conv3x3 / stride 1
        # No MaxPool
        # ----------------------------------------------------

        if small_input:

            self.backbone.conv1 = nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

            self.backbone.maxpool = nn.Identity()

        # ----------------------------------------------------
        # Replace classifier
        # ----------------------------------------------------

        in_features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Linear(
            in_features,
            num_classes,
        )

        # ----------------------------------------------------
        # Configure trainable layers
        # ----------------------------------------------------

        self.configure_trainable_layers()

    # ========================================================
    # Trainable Layers
    # ========================================================

    def configure_trainable_layers(self):

        # ----------------------------------------------------
        # Freeze everything first
        # ----------------------------------------------------

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # ----------------------------------------------------
        # R1
        #
        # Feature Extraction
        #
        # Only classifier is trainable.
        # ----------------------------------------------------

        if self.strategy == "feature_extraction":

            for parameter in (
                self.backbone.fc.parameters()
            ):
                parameter.requires_grad = True

        # ----------------------------------------------------
        # R2
        #
        # Fine-Tune Stage 1
        #
        # layer4 + classifier are trainable.
        # ----------------------------------------------------

        elif self.strategy == "fine_tune_stage_1":

            for parameter in (
                self.backbone.layer4.parameters()
            ):
                parameter.requires_grad = True

            for parameter in (
                self.backbone.fc.parameters()
            ):
                parameter.requires_grad = True

        # ----------------------------------------------------
        # R3
        #
        # Fine-Tune Stage 2
        #
        # layer3 + layer4 + classifier are trainable.
        # ----------------------------------------------------

        elif self.strategy == "fine_tune_stage_2":

            for parameter in (
                self.backbone.layer3.parameters()
            ):
                parameter.requires_grad = True

            for parameter in (
                self.backbone.layer4.parameters()
            ):
                parameter.requires_grad = True

            for parameter in (
                self.backbone.fc.parameters()
            ):
                parameter.requires_grad = True

        # ----------------------------------------------------
        # R4
        #
        # From Scratch
        #
        # Everything is trainable.
        # ----------------------------------------------------

        elif self.strategy == "from_scratch":

            for parameter in self.backbone.parameters():
                parameter.requires_grad = True

        else:

            raise ValueError(
                f"Unsupported strategy: "
                f"{self.strategy}"
            )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        return self.backbone(x)

    # ========================================================
    # Trainable Parameters
    # ========================================================

    def get_trainable_parameters(self):

        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]

    # ========================================================
    # Parameter Counts
    # ========================================================

    def get_trainable_parameter_count(self):

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def get_total_parameter_count(self):

        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )

    # ========================================================
    # Parameter Groups
    # ========================================================

    def get_parameter_groups(
        self,
        layer3_lr=1e-5,
        layer4_lr=1e-4,
        fc_lr=1e-3,
    ):

        # ----------------------------------------------------
        # R1
        # ----------------------------------------------------

        if self.strategy == "feature_extraction":

            fc_parameters = [
                parameter
                for parameter in (
                    self.backbone.fc.parameters()
                )
                if parameter.requires_grad
            ]

            return [
                {
                    "params": fc_parameters,
                    "lr": fc_lr,
                }
            ]

        # ----------------------------------------------------
        # R2
        # ----------------------------------------------------

        if self.strategy == "fine_tune_stage_1":

            layer4_parameters = [
                parameter
                for parameter in (
                    self.backbone.layer4.parameters()
                )
                if parameter.requires_grad
            ]

            fc_parameters = [
                parameter
                for parameter in (
                    self.backbone.fc.parameters()
                )
                if parameter.requires_grad
            ]

            return [
                {
                    "params": layer4_parameters,
                    "lr": layer4_lr,
                },
                {
                    "params": fc_parameters,
                    "lr": fc_lr,
                },
            ]

        # ----------------------------------------------------
        # R3
        # ----------------------------------------------------

        if self.strategy == "fine_tune_stage_2":

            layer3_parameters = [
                parameter
                for parameter in (
                    self.backbone.layer3.parameters()
                )
                if parameter.requires_grad
            ]

            layer4_parameters = [
                parameter
                for parameter in (
                    self.backbone.layer4.parameters()
                )
                if parameter.requires_grad
            ]

            fc_parameters = [
                parameter
                for parameter in (
                    self.backbone.fc.parameters()
                )
                if parameter.requires_grad
            ]

            return [
                {
                    "params": layer3_parameters,
                    "lr": layer3_lr,
                },
                {
                    "params": layer4_parameters,
                    "lr": layer4_lr,
                },
                {
                    "params": fc_parameters,
                    "lr": fc_lr,
                },
            ]

        # ----------------------------------------------------
        # R4
        # ----------------------------------------------------

        if self.strategy == "from_scratch":

            return [
                {
                    "params": self.get_trainable_parameters(),
                    "lr": fc_lr,
                }
            ]

        raise ValueError(
            f"Unsupported strategy: "
            f"{self.strategy}"
        )


# ============================================================
# Manual Test
# ============================================================

if __name__ == "__main__":

    strategies = [
        "feature_extraction",
        "fine_tune_stage_1",
        "fine_tune_stage_2",
        "from_scratch",
    ]

    for strategy in strategies:

        print("\n" + "=" * 70)
        print(f"Strategy: {strategy}")
        print("=" * 70)

        pretrained = (
            strategy != "from_scratch"
        )

        small_input = (
            strategy == "from_scratch"
        )

        image_size = (
            64
            if small_input
            else 224
        )

        model = ResNet18Model(
            num_classes=8,
            strategy=strategy,
            pretrained=pretrained,
            small_input=small_input,
        )

        print(
            f"Total parameters: "
            f"{model.get_total_parameter_count():,}"
        )

        print(
            f"Trainable parameters: "
            f"{model.get_trainable_parameter_count():,}"
        )

        dummy_input = torch.randn(
            2,
            3,
            image_size,
            image_size,
        )

        output = model(
            dummy_input
        )

        print(
            f"Input shape: "
            f"{dummy_input.shape}"
        )

        print(
            f"Output shape: "
            f"{output.shape}"
        )

        parameter_groups = (
            model.get_parameter_groups()
        )

        for index, group in enumerate(
            parameter_groups,
            start=1,
        ):

            parameter_count = sum(
                parameter.numel()
                for parameter in group["params"]
            )

            print(
                f"Group {index}: "
                f"LR={group['lr']:.1e}, "
                f"Params={parameter_count:,}"
            )