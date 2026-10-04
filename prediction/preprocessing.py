from torchvision import transforms
from PIL import Image, ImageOps


IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


class ResizeWithPadding:

    def __init__(
        self,
        size=(224, 224),
        fill=0,
    ):
        self.size = size
        self.fill = fill

    def __call__(self, image):

        target_width, target_height = self.size

        width, height = image.size

        scale = min(
            target_width / width,
            target_height / height,
        )

        new_width = round(width * scale)
        new_height = round(height * scale)

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR,
        )

        pad_left = (
            target_width - new_width
        ) // 2

        pad_top = (
            target_height - new_height
        ) // 2

        pad_right = (
            target_width
            - new_width
            - pad_left
        )

        pad_bottom = (
            target_height
            - new_height
            - pad_top
        )

        image = ImageOps.expand(
            image,
            border=(
                pad_left,
                pad_top,
                pad_right,
                pad_bottom,
            ),
            fill=self.fill,
        )

        return image


def get_eval_transform(
    image_size=(224, 224),
):

    if isinstance(image_size, int):
        image_size = (
            image_size,
            image_size,
        )

    return transforms.Compose(
        [
            ResizeWithPadding(
                size=image_size
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )