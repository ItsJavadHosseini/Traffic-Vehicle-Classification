from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd

from prediction.predictor import ModelPredictor


class BatchPredictor:

    def __init__(self, model_name, device=None):
        self.predictor = ModelPredictor(
            model_name=model_name,
            device=device,
        )

        self.model_name = model_name

    def predict_images(self, images):
        results = []

        for filename, image in images:
            try:
                result = self.predictor.predict(image)

                results.append(
                    {
                        "filename": filename,
                        "predicted_class": result["class_name"],
                        "confidence": result["confidence"],
                        "status": "ok",
                    }
                )

            except Exception as error:
                results.append(
                    {
                        "filename": filename,
                        "predicted_class": None,
                        "confidence": None,
                        "status": "error",
                        "error": str(error),
                    }
                )

        return pd.DataFrame(results)

    @staticmethod
    def create_zip(results, images):
        image_bytes = {
            filename: data
            for filename, data in images
        }

        zip_buffer = BytesIO()

        with zipfile.ZipFile(
            zip_buffer,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as zip_file:

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8-sig")

            zip_file.writestr(
                "predictions.csv",
                csv_data,
            )

            for _, row in results.iterrows():

                if row["status"] != "ok":
                    continue

                filename = row["filename"]
                predicted_class = row["predicted_class"]

                if filename not in image_bytes:
                    continue

                zip_path = (
                    Path(predicted_class)
                    / filename
                )

                zip_file.writestr(
                    str(zip_path),
                    image_bytes[filename],
                )

        zip_buffer.seek(0)

        return zip_buffer.getvalue()