import streamlit as st
from pathlib import Path
from PIL import Image
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference import InferenceEngine
from models.unet_densenet import CLASSES

st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

st.title("Brain Tumor MRI Classification with Grad-CAM")

weights_path = "models/weights/best.pt"

uploaded = st.file_uploader("Upload MRI image (jpg/png)", type=["jpg", "jpeg", "png"]) 

if 'engine' not in st.session_state and Path(weights_path).exists():
    try:
        st.session_state.engine = InferenceEngine(Path(weights_path), device='cpu', enable_cam=True)
    except Exception as e:
        st.error(f"Failed to load model: {e}")

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", width='stretch')
    if 'engine' not in st.session_state:
        st.warning("Model not loaded yet or weights missing.")
    else:
        with st.spinner("Running inference..."):
            result = st.session_state.engine.predict(image, return_cam=True)
        st.subheader(f"Prediction: {result['class']} ({result['confidence']*100:.1f}% confidence)")
        if 'heatmap' in result:
            st.image(result['heatmap'], caption="Grad-CAM Heatmap", width='stretch')
        st.write("Probabilities:")
        for c in CLASSES:
            st.write(f"- {c}: {result['probs'][c]*100:.2f}%")
