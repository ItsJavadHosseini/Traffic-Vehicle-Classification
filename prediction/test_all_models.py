from PIL import Image

from prediction.model_registry import MODEL_REGISTRY
from prediction.predictor import ModelPredictor


image_path = r"D:\Desktop\Traffic-Vehicle-Classification\202045377.jpg"

image = Image.open(image_path)


for model_name in MODEL_REGISTRY:

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    predictor = ModelPredictor(model_name)

    result = predictor.predict(image)

    print(f"Class      : {result['class_name']}")
    print(f"Class index: {result['class_index']}")
    print(f"Confidence : {result['confidence']:.4f}")