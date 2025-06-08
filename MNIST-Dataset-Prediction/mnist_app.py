import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
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
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), mode='RGBA').convert('L')

    img = ImageOps.invert(img)

    img = img.resize((28, 28))

    img_array = np.array(img).astype('float32') / 255.0

    img_array = img_array.reshape(1, 28, 28, 1)

    st.subheader("Processed Input (28x28 Grayscale)")
    st.image(img.resize((140, 140)), caption="Input to Model", width=150)

    if st.button("Predict"):
        prediction = model.predict(img_array)
        pred_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100
        st.success(f"✅ Predicted Digit: **{pred_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
