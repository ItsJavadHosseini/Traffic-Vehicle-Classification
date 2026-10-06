import streamlit as st
from PIL import Image

from prediction.batch_predictor import BatchPredictor
from prediction.evaluator import ModelEvaluator
from prediction.model_registry import MODEL_REGISTRY
from prediction.predictor import ModelPredictor


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Traffic Vision",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background-color: #0b0b0b;
        color: #ffffff;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Header ---------- */

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }

    .main-title span {
        color: #ff7a00;
    }

    .subtitle {
        color: #a6a6a6;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* ---------- Labels ---------- */

    label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* ---------- Selectbox ---------- */

    div[data-baseweb="select"] > div {
        background-color: #151515;
        border: 1px solid #333333;
        border-radius: 10px;
    }

    /* ---------- File uploader ---------- */

    section[data-testid="stFileUploader"] {
        background-color: #121212;
        border: 1px dashed #444444;
        border-radius: 12px;
        padding: 1rem;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        background-color: #ff7a00;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        min-height: 42px;
    }

    .stButton > button:hover {
        background-color: #ff8f26;
        color: #ffffff;
    }

    .stDownloadButton > button {
        background-color: #151515;
        color: #ff7a00;
        border: 1px solid #ff7a00;
        border-radius: 8px;
        font-weight: 600;
    }

    .stDownloadButton > button:hover {
        background-color: #ff7a00;
        color: #ffffff;
    }

    /* ---------- Radio / Mode ---------- */

    div[role="radiogroup"] {
        gap: 0.35rem;
    }

    div[role="radiogroup"] label {
        background-color: #151515;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 0.35rem 0.8rem;
        color: #cccccc !important;
    }

    div[role="radiogroup"] label:hover {
        border-color: #ff7a00;
    }

    /* ---------- Cards ---------- */

    .metric-card {
        background-color: #151515;
        border: 1px solid #292929;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .metric-label {
        color: #999999;
        font-size: 0.85rem;
    }

    .metric-value {
        color: #ff7a00;
        font-size: 1.5rem;
        font-weight: 800;
    }

    /* ---------- Divider ---------- */

    hr {
        border-color: #292929;
    }

    /* ---------- Dataframe ---------- */

    div[data-testid="stDataFrame"] {
        border: 1px solid #292929;
        border-radius: 10px;
    }

    /* ---------- Info ---------- */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# HEADER
# ==================================================

st.markdown(
    """
    <div class="main-title">
        Traffic <span>Vision</span>
    </div>

    <div class="subtitle">
        Traffic Vehicle Classification & Model Evaluation
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# MODEL + MODE
# ==================================================

col_model, col_mode = st.columns(
    [1.3, 2.7]
)


with col_model:

    model_name = st.selectbox(
        "Model",
        options=list(
            MODEL_REGISTRY.keys()
        ),
    )


with col_mode:

    mode = st.radio(
        "Mode",
        [
            "Single Image",
            "Batch Prediction",
            "Evaluation",
        ],
        horizontal=True,
    )


st.divider()


# ==================================================
# SINGLE IMAGE
# ==================================================

if mode == "Single Image":

    st.subheader("Single Image Prediction")

    uploaded_file = st.file_uploader(
        "Upload image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp",
        ],
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption=uploaded_file.name,
            width="stretch",
        )

        if st.button(
            "Predict",
            type="primary",
        ):

            with st.spinner(
                "Running prediction..."
            ):

                predictor = ModelPredictor(
                    model_name=model_name
                )

                result = predictor.predict(
                    image
                )

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "### Predicted Class"
                )

                st.markdown(
                    f"## :orange[{result['class_name']}]"
                )

            with col2:

                st.markdown(
                    "### Confidence"
                )

                st.markdown(
                    f"## :orange[{result['confidence']:.2%}]"
                )


# ==================================================
# BATCH PREDICTION
# ==================================================

elif mode == "Batch Prediction":

    st.subheader("Batch Prediction")

    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp",
        ],
        accept_multiple_files=True,
    )

    if uploaded_files:

        st.info(
            f"{len(uploaded_files)} images selected."
        )

        if st.button(
            "Predict All",
            type="primary",
        ):

            images_for_prediction = []
            original_images = []

            for uploaded_file in uploaded_files:

                image_bytes = (
                    uploaded_file.getvalue()
                )

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                images_for_prediction.append(
                    (
                        uploaded_file.name,
                        image,
                    )
                )

                original_images.append(
                    (
                        uploaded_file.name,
                        image_bytes,
                    )
                )

            with st.spinner(
                "Predicting images..."
            ):

                batch_predictor = BatchPredictor(
                    model_name=model_name
                )

                results = (
                    batch_predictor.predict_images(
                        images_for_prediction
                    )
                )

            st.success(
                "Batch prediction completed."
            )

            st.dataframe(
                results,
                width="stretch",
            )

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8-sig")

            col1, col2 = st.columns(2)

            with col1:

                st.download_button(
                    "Download CSV",
                    data=csv_data,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

            with col2:

                zip_data = (
                    batch_predictor.create_zip(
                        results,
                        original_images,
                    )
                )

                st.download_button(
                    "Download ZIP",
                    data=zip_data,
                    file_name="predictions.zip",
                    mime="application/zip",
                )


# ==================================================
# EVALUATION
# ==================================================

else:

    st.subheader("Model Evaluation")

    st.write(
        "Evaluate the selected model against a labeled dataset."
    )

    st.info(
        "Each subfolder must represent one true class."
    )

    dataset_path = st.text_input(
        "Test Dataset Folder",
        placeholder=(
            r"D:\Desktop\vehicle_test"
        ),
    )

    if st.button(
        "Evaluate Model",
        type="primary",
    ):

        if not dataset_path:

            st.warning(
                "Please enter the test dataset path."
            )

        else:

            with st.spinner(
                "Evaluating model..."
            ):

                evaluator = ModelEvaluator(
                    model_name=model_name
                )

                evaluation = evaluator.evaluate(
                    dataset_path
                )

            results = evaluation["results"]
            metrics = evaluation["metrics"]

            st.success(
                "Evaluation completed."
            )

            # --------------------------------------
            # Main Metrics
            # --------------------------------------

            st.subheader("Performance")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Accuracy",
                f"{metrics['accuracy']:.2%}",
            )

            col2.metric(
                "Macro F1",
                f"{metrics['macro_f1']:.4f}",
            )

            col3.metric(
                "Precision",
                f"{metrics['macro_precision']:.4f}",
            )

            col4.metric(
                "Recall",
                f"{metrics['macro_recall']:.4f}",
            )

            # --------------------------------------
            # Confidence
            # --------------------------------------

            st.subheader(
                "Confidence Analysis"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Average",
                f"{metrics['average_confidence']:.2%}",
            )

            col2.metric(
                "Correct",
                f"{metrics['correct_confidence']:.2%}",
            )

            col3.metric(
                "Wrong",
                f"{metrics['wrong_confidence']:.2%}",
            )

            # --------------------------------------
            # Dataset Stats
            # --------------------------------------

            st.subheader(
                "Dataset Statistics"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Images",
                metrics["total_images"],
            )

            col2.metric(
                "Successful",
                metrics["successful_predictions"],
            )

            col3.metric(
                "Failed",
                metrics["failed_predictions"],
            )

            st.divider()

            # --------------------------------------
            # Classification Report
            # --------------------------------------

            st.subheader(
                "Classification Report"
            )

            report = (
                evaluator.get_classification_report(
                    results
                )
            )

            st.code(report)

            # --------------------------------------
            # Confusion Matrix
            # --------------------------------------

            st.subheader(
                "Confusion Matrix"
            )

            confusion = (
                evaluator.get_confusion_matrix(
                    results
                )
            )

            confusion_df = confusion

            st.dataframe(
                confusion_df,
                width="stretch",
            )

            # --------------------------------------
            # Detailed Results
            # --------------------------------------

            st.subheader(
                "Detailed Predictions"
            )

            st.dataframe(
                results.drop(
                    columns=["source_path"],
                    errors="ignore",
                ),
                width="stretch",
            )

            # --------------------------------------
            # Evaluation ZIP
            # --------------------------------------

            st.divider()

            st.subheader(
                "Export Evaluation"
            )

            zip_data = (
                evaluator.create_zip(
                    results
                )
            )

            st.download_button(
                "Download Evaluation ZIP",
                data=zip_data,
                file_name="evaluation_results.zip",
                mime="application/zip",
            )