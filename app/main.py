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

st.title("Brain Tumor MRI Classification with DenseNet121 and with Grad-CAM Heatmaps")

st.write("""
This app uses a post-trained DenseNet121 machine learning model [(Huang et al., 2017)](https://arxiv.org/abs/1608.06993) to classify brain MRI images into four categories: Glioma, Meningioma, Pituitary tumors, or No Tumor. 
It also generates Grad-CAM heatmaps to visualize areas influencing the model's decisions. 

The model was post-trained on the Figshare 2024 brain tumor dataset [(Afzal, 2024)](https://figshare.com/articles/figure/Brain_tumor_dataset/27102082).
""")

st.write("## Disclaimer")
st.write("""
This app is purely a proof of concept. It is not a substitute for professional medical advice, diagnosis, or treatment. 
Always seek the advice of your physician or other qualified healthcare professional with any questions you may have regarding a medical condition.
""")
uploaded = st.file_uploader("Upload MRI image (jpg/png)", type=["jpg", "jpeg", "png"]) 

weights_path = "models/weights/best.pt"

if 'engine' not in st.session_state and Path(weights_path).exists():
    try:
        st.session_state.engine = InferenceEngine(Path(weights_path), device='cpu', enable_cam=True)
    except Exception as e:
        st.error(f"Failed to load model: {e}")

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", width='stretch')
    if 'engine' not in st.session_state:
        st.warning("Model not loaded yet or weights missing. Please refer to the README for setup instructions.")
    else:
        with st.spinner("Running inference..."):
            result = st.session_state.engine.predict(image, return_cam=True)
        st.subheader(f"Prediction: {result['class']} ({result['confidence']*100:.1f}% confidence)")
        if 'heatmap' in result:
            st.image(result['heatmap'], caption="Grad-CAM Heatmap", width='stretch')
        st.write("Probabilities:")
        for c in CLASSES:
            st.write(f"- {c}: {result['probs'][c]*100:.2f}%")

st.write("### Glossary")
st.write("""
         
__MRI__: Magnetic Resonance Imaging, a medical imaging technique used to visualize detailed internal structures, particularly soft tissues like the brain.

__DenseNet121__: A pre-trained convolutional neural network used as the base model (backbone) for feature extraction and classification in this app.

__PyTorch__: An open-source machine learning library used for applications such as computer vision and natural language processing.

__Grad-CAM__: Gradient-weighted Class Activation Mapping, an explainable AI (XAI) technique for producing visual explanations for decisions from convolutional neural networks.

__Heatmap__: A graphical representation of data where individual values are represented as colors, 
often used to visualize areas of interest in images. Here, it highlights regions in the MRI that influenced the model's prediction in red.
         
__Streamlit__: An open-source python app framework used to create and share data apps without requiring web development skills.
         
#### Tumor Types:
- **Glioma**: A type of tumor that occurs in the brain and spinal cord, originating from glial cells.
- **Meningioma**: A tumor that arises from the meninges, the membranes that surround the brain and spinal cord.
- **Pituitary Tumor**: A growth in the pituitary gland, which can affect hormone levels and bodily functions.
- **No Tumor**: Indicates that the MRI scan does not show any signs of a tumor.
""")

st.caption("Made by [Rubin James](https://github.com/chempananickal) (BIN23 Group 3) for the *Projektpraktikum* course at the [Provadis School of International Management & Technology](https://www.provadis-hochschule.de) 2025.")