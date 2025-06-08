import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load trained model
model = load_model("mnist_cnn_model.keras")

st.title("MNIST Digit Recognizer")
st.write("Draw a digit (0–9) below:")

# Canvas settings
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=12,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    tensorflow.compat.v1.reset_default_graph()
    img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype(np.uint8))
    img = img.resize((28, 28))
    img = ImageOps.invert(img)

    st.subheader("Model Input Preview")
    st.image(img.resize((140, 140)), caption="28x28 Grayscale")

    img = np.array(img).reshape(1, 28, 28, 1).astype("float32") / 255.0

    # Predict
    if st.button("Predict"):
        prediction = model.predict(img)
        pred_class = np.argmax(prediction)
        st.success(f"Predicted Digit: {pred_class}")
