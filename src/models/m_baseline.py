import torch
import torch.nn as nn


class DeepBaselineCNN(nn.Module):

    def __init__(self, num_classes=8, dropout=0.3, pooling="max"):
        super().__init__()

        if pooling not in {"max", "avg"}:
            raise ValueError(
                f"Unsupported pooling: {pooling}. "
                f"Choose 'max' or 'avg'."
            )

        def make_pool():
            if pooling == "max":
                return nn.MaxPool2d(kernel_size=2)
            return nn.AvgPool2d(kernel_size=2)

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            make_pool(),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            make_pool(),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            make_pool(),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            make_pool(),
        )

        # Final spatial aggregation
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":

    for pooling in ["max", "avg"]:

        print("\n" + "=" * 60)
        print(f"Pooling: {pooling}")
        print("=" * 60)

        model = DeepBaselineCNN(
            num_classes=8,
            dropout=0.3,
            pooling=pooling
        )

        total_params = sum(
            p.numel()
            for p in model.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)

        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Input shape:          {dummy_input.shape}")
        print(f"Output shape:         {output.shape}")