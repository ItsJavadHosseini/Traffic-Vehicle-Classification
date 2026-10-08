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

    /* ==================================================
       GLOBAL
       ================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(0, 255, 140, 0.06),
                transparent 30%
            ),
            #070a08;
        color: #f2fff8;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ==================================================
       HEADER
       ================================================== */

    .main-title {
        font-size: 2.6rem;
        font-weight: 850;
        color: #f2fff8;
        margin-bottom: 0.15rem;
        letter-spacing: -1px;
    }

    .main-title span {
        color: #00e676;
    }

    .subtitle {
        color: #7f9589;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    /* ==================================================
       LABELS
       ================================================== */

    label {
        color: #e9fff2 !important;
        font-weight: 600 !important;
    }

    /* ==================================================
       SELECTBOX
       ================================================== */

    div[data-baseweb="select"] > div {
        background-color: #0e1511;
        border: 1px solid #24352b;
        border-radius: 10px;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: #00e676;
    }

    /* ==================================================
       FILE UPLOADER
       ================================================== */

    section[data-testid="stFileUploader"] {
        background-color: #0c120f;
        border: 1px dashed #31513e;
        border-radius: 12px;
        padding: 1rem;
    }

    section[data-testid="stFileUploader"]:hover {
        border-color: #00e676;
    }

    /* ==================================================
       BUTTONS
       ================================================== */

    .stButton > button {
        background: linear-gradient(
            135deg,
            #00e676,
            #00b85c
        );
        color: #031008;
        border: none;
        border-radius: 9px;
        font-weight: 800;
        min-height: 42px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(
            135deg,
            #19ff91,
            #00d86b
        );
        color: #031008;
        transform: translateY(-1px);
        box-shadow:
            0 6px 20px rgba(0, 230, 118, 0.18);
    }

    .stDownloadButton > button {
        background-color: #0d1511;
        color: #00e676;
        border: 1px solid #00a957;
        border-radius: 9px;
        font-weight: 700;
    }

    .stDownloadButton > button:hover {
        background-color: #00e676;
        color: #031008;
        border-color: #00e676;
    }

    /* ==================================================
       RADIO / MODE
       ================================================== */

    div[role="radiogroup"] {
        gap: 0.35rem;
    }

    div[role="radiogroup"] label {
        background-color: #0e1511;
        border: 1px solid #24352b;
        border-radius: 8px;
        padding: 0.35rem 0.8rem;
        color: #a8baaf !important;
    }

    div[role="radiogroup"] label:hover {
        border-color: #00e676;
    }

    /* ==================================================
       METRIC CARDS
       ================================================== */

    .metric-card {
        background:
            linear-gradient(
                145deg,
                #101a14,
                #0b110e
            );
        border: 1px solid #20372a;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .metric-label {
        color: #7f9589;
        font-size: 0.85rem;
    }

    .metric-value {
        color: #00e676;
        font-size: 1.5rem;
        font-weight: 800;
    }

    /* ==================================================
       DIVIDER
       ================================================== */

    hr {
        border-color: #1d2d23;
    }

    /* ==================================================
       DATAFRAME
       ================================================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid #20372a;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ==================================================
       ALERTS
       ================================================== */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* ==================================================
       OOD WARNING
       ================================================== */

    .status-unknown {
        background: rgba(255, 193, 7, 0.08);
        border: 1px solid rgba(255, 193, 7, 0.35);
        color: #ffc107;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-weight: 700;
    }

    /* ==================================================
       RESULT BOX
       ================================================== */

    .result-box {
        background:
            linear-gradient(
                145deg,
                #101b14,
                #0b110e
            );
        border: 1px solid #203b2b;
        border-radius: 14px;
        padding: 1.2rem;
        margin-top: 1rem;
    }

    .result-label {
        color: #718579;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .result-value {
        color: #f2fff8;
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    .result-value-green {
        color: #00e676;
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    /* ==================================================
       IMAGE
       ================================================== */

    img {
        border-radius: 12px;
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

col_model, col_mode = st.columns([1.3, 2.7])


with col_model:

    model_name = st.selectbox(
        "Model",
        options=list(MODEL_REGISTRY.keys()),
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

            # --------------------------------------
            # Main Result
            # --------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "### Final Prediction"
                )

                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="result-label">
                            Classification Result
                        </div>
                        <div class="result-value-green">
                            {result["class_name"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:

                st.markdown(
                    "### Confidence"
                )

                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="result-label">
                            Softmax Confidence
                        </div>
                        <div class="result-value-green">
                            {result["confidence"]:.2%}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")

            # --------------------------------------
            # OOD Analysis
            # --------------------------------------

            st.subheader("OOD Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="result-label">
                            Raw Prediction
                        </div>
                        <div class="result-value">
                            {result["raw_class_name"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:

                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="result-label">
                            Energy Score
                        </div>
                        <div class="result-value">
                            {result["energy"]:.4f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col3:

                threshold = result.get(
                    "energy_threshold"
                )

                if threshold is not None:

                    st.markdown(
                        f"""
                        <div class="result-box">
                            <div class="result-label">
                                Energy Threshold
                            </div>
                            <div class="result-value">
                                {threshold:.4f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                else:

                    st.markdown(
                        """
                        <div class="result-box">
                            <div class="result-label">
                                Energy Threshold
                            </div>
                            <div class="result-value">
                                N/A
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.write("")

            # --------------------------------------
            # OOD Warning
            # --------------------------------------

            if result["is_unknown"]:

                st.markdown(
                    """
                    <div class="status-unknown">
                        ⚠️ Possible Unknown Class — Please review manually.
                    </div>
                    """,
                    unsafe_allow_html=True,
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

            st.dataframe(
                confusion,
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