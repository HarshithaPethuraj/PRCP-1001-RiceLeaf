"""
Rice Leaf Disease Detector — Hugging Face Space (Streamlit)
============================================================
Model: MobileNetV2 transfer learning, trained on the PRCP-1001 Rice Leaf dataset.
Author: Harshitha Pethuraj
"""

import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import streamlit as st
from PIL import Image
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEIGHTS_PATH = "rice_leaf_weights.npy"
IMG_SIZE     = (224, 224)
CLASSES      = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]

TREATMENT_TIPS = {
    "Bacterial leaf blight": (
        "**Bacterial leaf blight (Xanthomonas oryzae)**\n\n"
        "- Use disease-resistant rice varieties for the next season.\n"
        "- Drain the field briefly to suppress bacterial spread.\n"
        "- Avoid excess nitrogen fertilizer — it accelerates blight.\n"
        "- Apply copper-based bactericides only if recommended by a local agronomist.\n"
        "- Remove and destroy infected plant debris after harvest."
    ),
    "Brown spot": (
        "**Brown spot (Bipolaris oryzae / Cochliobolus miyabeanus)**\n\n"
        "- Often caused by potassium or silicon deficiency — soil-test and correct.\n"
        "- Use clean, certified seed; treat seeds with a fungicide before sowing.\n"
        "- Maintain balanced fertilization, especially potassium.\n"
        "- Spray a recommended fungicide (e.g. mancozeb, propiconazole) at early symptom stage.\n"
        "- Rotate crops to break the disease cycle."
    ),
    "Leaf smut": (
        "**Leaf smut (Entyloma oryzae)**\n\n"
        "- Usually a minor disease — rarely causes major yield loss on its own.\n"
        "- Use disease-free seed and resistant varieties where available.\n"
        "- Avoid excessive nitrogen application.\n"
        "- Improve field drainage and remove infected stubble after harvest.\n"
        "- Apply fungicides only if infection is widespread."
    ),
}

DISCLAIMER = (
    "⚠️ **Disclaimer:** This is an academic/demo tool. Always confirm any diagnosis "
    "with a local agricultural extension officer before applying pesticides or fungicides."
)


def preprocess(arr):
    return (arr.astype(np.float32) / 127.5) - 1.0


def build_model():
    base = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    base.trainable = False
    x = base.output
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = keras.layers.Dropout(0.3, name="dropout")(x)
    out = keras.layers.Dense(3, activation="softmax", name="dense")(x)
    model = keras.Model(inputs=base.input, outputs=out)
    return model


@st.cache_resource
def load_model_and_layer():
    model = build_model()
    weights = list(np.load(WEIGHTS_PATH, allow_pickle=True))
    model.set_weights(weights)

    last_conv = None
    for cand in ["Conv_1", "out_relu", "top_conv"]:
        try:
            model.get_layer(cand)
            last_conv = cand
            break
        except Exception:
            continue
    if last_conv is None:
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                last_conv = layer.name
                break
    return model, last_conv


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def compute_gradcam(img_array, model, last_conv_name):
    grad_model = keras.Model(
        model.inputs,
        [model.get_layer(last_conv_name).output, model.output],
    )
    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_out, preds = grad_model(img_tensor)
        cls  = tf.argmax(preds[0])
        loss = preds[:, cls]
    grads  = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heat = conv_out @ pooled[..., None]
    heat = tf.squeeze(heat)
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy(), int(cls)


def overlay_heatmap(original_pil, heat, alpha=0.45):
    img          = np.asarray(original_pil.resize(IMG_SIZE)).astype(np.float32)
    heat_resized = tf.image.resize(heat[..., None], IMG_SIZE).numpy().squeeze()
    colored      = cm.jet(heat_resized)[..., :3] * 255.0
    blended      = (1 - alpha) * img + alpha * colored
    blended      = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Rice Leaf Disease Detector", page_icon="🌾", layout="wide")
st.title("🌾 Rice Leaf Disease Detector")

st.markdown(
    """
    Upload a photo of a rice leaf and the model will:
    1. **Predict the disease** from 3 classes — Bacterial leaf blight, Brown spot, or Leaf smut
    2. **Show a Grad-CAM heatmap** of which leaf regions drove the prediction
    3. **Suggest a treatment** for the most likely disease

    Built with MobileNetV2 transfer learning on the PRCP-1001 dataset.
    Validation accuracy ≈ 93%, macro-F1 ≈ 0.93, macro ROC-AUC ≈ 0.997.
    """
)

with st.spinner("Loading model..."):
    model, last_conv = load_model_and_layer()

uploaded = st.file_uploader("Upload a rice-leaf image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil   = Image.open(uploaded).convert("RGB")
    arr   = np.asarray(pil.resize(IMG_SIZE), dtype=np.float32)
    arr   = preprocess(arr)
    batch = np.expand_dims(arr, 0)

    with st.spinner("Analyzing..."):
        probs   = model.predict(batch, verbose=0)[0]
        heat, _ = compute_gradcam(batch, model, last_conv)
        overlay = overlay_heatmap(pil, heat)

    top_class = CLASSES[int(np.argmax(probs))]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your image")
        st.image(pil, use_column_width=True)
    with col2:
        st.subheader("Grad-CAM — what the model looked at")
        st.image(overlay, use_column_width=True)

    st.subheader("Top-3 predictions")
    for i in np.argsort(probs)[::-1]:
        conf = float(probs[i])
        st.write(f"**{CLASSES[i]}** — {conf*100:.1f}%")
        st.progress(conf)

    st.subheader(f"Suggested treatment for: {top_class}")
    st.markdown(TREATMENT_TIPS[top_class])
    st.info(DISCLAIMER)
else:
    st.info("👆 Upload a rice-leaf image to get started.")

st.markdown("---")
st.markdown(
    """
    **Project 4** · Agriculture Domain · Rice Leaf Disease Image Dataset
    **Project Type** · Deep Learning | Image Classification using Transfer Learning (CNN)
    **Author** · Harshitha Pethuraj · **Team ID** · PTID-AIE-MAY-26-11194
    """
)