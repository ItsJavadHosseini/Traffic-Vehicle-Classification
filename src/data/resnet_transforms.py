
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_resnet_train_transform(image_size=224, augmentation=None):
    """
    Build the training transform.

    If augmentation is disabled or omitted, preserve the original
    preprocessing pipeline.
    """
    augmentation = augmentation or {}

    if not augmentation.get("enabled", False):
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ])

    train_transforms = []

    random_crop = augmentation.get("random_resized_crop", {})
    if random_crop:
        train_transforms.append(
            transforms.RandomResizedCrop(
                size=image_size,
                scale=tuple(random_crop.get("scale", [0.8, 1.0])),
                ratio=tuple(random_crop.get("ratio", [0.9, 1.1])),
            )
        )
    else:
        train_transforms.extend([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ])

    if augmentation.get("horizontal_flip", False):
        train_transforms.append(
            transforms.RandomHorizontalFlip(p=0.5)
        )

    rotation = augmentation.get("rotation", 0)
    if rotation:
        train_transforms.append(
            transforms.RandomRotation(degrees=rotation)
        )

    color_jitter = augmentation.get("color_jitter", {})
    if color_jitter:
        train_transforms.append(
            transforms.ColorJitter(
                brightness=color_jitter.get("brightness", 0),
                contrast=color_jitter.get("contrast", 0),
                saturation=color_jitter.get("saturation", 0),
                hue=color_jitter.get("hue", 0),
            )
        )

    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    return transforms.Compose(train_transforms)


def get_resnet_eval_transform(image_size=224):
    """
    Deterministic preprocessing for validation and evaluation.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])
