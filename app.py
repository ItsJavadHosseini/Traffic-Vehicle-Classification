import streamlit as st
from PIL import Image

from prediction.model_registry import MODEL_REGISTRY
from prediction.predictor import ModelPredictor


st.set_page_config(
    page_title="Traffic Vehicle Classification",
    page_icon="🚗",
    layout="centered",
)


st.title("Traffic Vehicle Classification")
st.write("Upload an image and select a trained model.")


# ---------------------------------------------------------
# Model selection
# ---------------------------------------------------------

model_name = st.selectbox(
    "Select Model",
    options=list(MODEL_REGISTRY.keys()),
)


# ---------------------------------------------------------
# Image upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch",
    )

    if st.button("Predict"):

        with st.spinner("Loading model and predicting..."):

            predictor = ModelPredictor(
                model_name=model_name
            )

            result = predictor.predict(image)

        st.divider()

        st.subheader("Prediction")

        st.write(
            f"**Class:** {result['class_name']}"
        )

        st.write(
            f"**Confidence:** "
            f"{result['confidence']:.2%}"
        )