import datetime
import io

import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("Digit Recognizer")

# Canvas for drawing
drawing_mode = "freedraw"
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode=drawing_mode,
    key="canvas",
)


# Preprocess the image for prediction
def preprocess(img_data):
    if img_data is None:
        return None

    # Convert to grayscale by taking the mean of RGB channels
    img = np.mean(img_data[:, :, :3], axis=2)

    # Normalize to [0, 1]
    img = img / 255.0

    # Convert to PIL Image for proper resizing
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized) / 255.0

    return img_array.tolist()


# Sidebar for API URL
api_url = st.sidebar.text_input("FastAPI URL", "http://localhost:8000/predict")

# Prediction and UI
prediction = None
confidence = None
img_array = (
    preprocess(canvas_result.image_data)
    if canvas_result.image_data is not None
    else None
)

if st.button("Predict") and img_array is not None:
    with st.spinner("Predicting..."):
        response = requests.post(api_url, json={"image": img_array})
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            confidence = result["confidence"]
        else:
            st.error(f"Prediction failed: {response.text}")

if prediction is not None:
    st.markdown(f"**Prediction:** {prediction}")
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")

# True label input
true_label = st.number_input(
    "True label (0-9)",
    min_value=0,
    max_value=9,
    step=1,
    value=prediction if prediction is not None else 0,
)

# History table (stored in session state)
if "history" not in st.session_state:
    st.session_state["history"] = []

if st.button("Submit") and prediction is not None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["history"].insert(
        0, {"timestamp": timestamp, "pred": prediction, "label": true_label}
    )
    # Limit history to 50 rows
    st.session_state["history"] = st.session_state["history"][:50]

# Display history
table = st.session_state["history"]
if table:
    st.markdown("### History")
    st.table(table)
