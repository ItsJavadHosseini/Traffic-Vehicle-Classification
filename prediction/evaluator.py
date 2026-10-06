from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from prediction.predictor import ModelPredictor


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


class ModelEvaluator:

    def __init__(self, model_name, device=None):
        self.predictor = ModelPredictor(
            model_name=model_name,
            device=device,
        )

        self.model_name = model_name
        self.class_names = self.predictor.class_names

    def evaluate(self, dataset_path):

        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{dataset_path}"
            )

        results = []

        for class_dir in sorted(dataset_path.iterdir()):

            if not class_dir.is_dir():
                continue

            true_class = class_dir.name

            for image_path in sorted(class_dir.iterdir()):

                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                try:
                    from PIL import Image

                    image = Image.open(
                        image_path
                    ).convert("RGB")

                    prediction = self.predictor.predict(
                        image
                    )

                    predicted_class = (
                        prediction["class_name"]
                    )

                    confidence = (
                        prediction["confidence"]
                    )

                    results.append(
                        {
                            "filename": image_path.name,
                            "true_class": true_class,
                            "predicted_class": predicted_class,
                            "confidence": confidence,
                            "correct": (
                                true_class == predicted_class
                            ),
                            "status": "ok",
                            "source_path": str(image_path),
                        }
                    )

                except Exception as error:

                    results.append(
                        {
                            "filename": image_path.name,
                            "true_class": true_class,
                            "predicted_class": None,
                            "confidence": None,
                            "correct": False,
                            "status": "error",
                            "error": str(error),
                            "source_path": str(image_path),
                        }
                    )

        results_df = pd.DataFrame(results)

        metrics = self._calculate_metrics(
            results_df
        )

        return {
            "results": results_df,
            "metrics": metrics,
        }

    def _calculate_metrics(self, results):

        valid_results = results[
            results["status"] == "ok"
        ]

        if valid_results.empty:
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "weighted_f1": 0.0,
                "average_confidence": 0.0,
                "correct_confidence": 0.0,
                "wrong_confidence": 0.0,
                "total_images": len(results),
                "successful_predictions": 0,
                "failed_predictions": len(results),
            }

        y_true = valid_results["true_class"]
        y_pred = valid_results["predicted_class"]

        confidence = valid_results["confidence"]
        correct = valid_results["correct"]

        return {
            "accuracy": accuracy_score(
                y_true,
                y_pred,
            ),

            "macro_f1": f1_score(
                y_true,
                y_pred,
                labels=self.class_names,
                average="macro",
                zero_division=0,
            ),

            "macro_precision": precision_score(
                y_true,
                y_pred,
                labels=self.class_names,
                average="macro",
                zero_division=0,
            ),

            "macro_recall": recall_score(
                y_true,
                y_pred,
                labels=self.class_names,
                average="macro",
                zero_division=0,
            ),

            "weighted_f1": f1_score(
                y_true,
                y_pred,
                labels=self.class_names,
                average="weighted",
                zero_division=0,
            ),

            "average_confidence": confidence.mean(),

            "correct_confidence": (
                confidence[correct].mean()
                if correct.any()
                else 0.0
            ),

            "wrong_confidence": (
                confidence[~correct].mean()
                if (~correct).any()
                else 0.0
            ),

            "total_images": len(results),

            "successful_predictions": len(
                valid_results
            ),

            "failed_predictions": (
                len(results)
                - len(valid_results)
            ),
        }

    def get_confusion_matrix(self, results):

        valid_results = results[
            results["status"] == "ok"
        ]

        return confusion_matrix(
            valid_results["true_class"],
            valid_results["predicted_class"],
            labels=self.class_names,
        )

    def get_classification_report(self, results):

        valid_results = results[
            results["status"] == "ok"
        ]

        return classification_report(
            valid_results["true_class"],
            valid_results["predicted_class"],
            labels=self.class_names,
            zero_division=0,
        )

    @staticmethod
    def create_zip(results):

        zip_buffer = BytesIO()

        with zipfile.ZipFile(
            zip_buffer,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as zip_file:

            # ----------------------------------
            # CSV
            # ----------------------------------

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8-sig")

            zip_file.writestr(
                "evaluation_results.csv",
                csv_data,
            )

            # ----------------------------------
            # Images
            # ----------------------------------

            valid_results = results[
                results["status"] == "ok"
            ]

            for _, row in valid_results.iterrows():

                source_path = Path(
                    row["source_path"]
                )

                if not source_path.exists():
                    continue

                predicted_class = (
                    row["predicted_class"]
                )

                true_class = row["true_class"]

                filename = row["filename"]

                # Images are grouped by predicted class.
                #
                # predicted/
                # ├── ambulance/
                # ├── savari/
                # └── taxi/

                zip_path = (
                    Path("predicted")
                    / predicted_class
                    / filename
                )

                zip_file.write(
                    source_path,
                    str(zip_path),
                )

        zip_buffer.seek(0)

        return zip_buffer.getvalue()