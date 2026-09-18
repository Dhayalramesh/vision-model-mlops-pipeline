"""
MLOps Experiment Tracking Dashboard
Displays MLflow-logged experiment runs and serves the best model for live inference.
"""

import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os

st.set_page_config(page_title="Vision Model MLOps Pipeline", layout="wide")

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


# ---------- Model definition (must match training script) ----------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource
def load_model():
    model = SimpleCNN()
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model


@st.cache_data
def load_runs():
    if os.path.exists("runs_summary.csv"):
        return pd.read_csv("runs_summary.csv")
    return pd.DataFrame()


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ---------------- UI ----------------
st.title("Vision Model MLOps Pipeline")
st.caption("Experiment runs logged with MLflow · Best model auto-selected by validation accuracy")

runs_df = load_runs()
model = load_model()

tab1, tab2 = st.tabs(["Experiment Comparison", "Live Inference"])

with tab1:
    if runs_df.empty:
        st.warning("No runs_summary.csv found. Run the training notebook first and add it to this repo.")
    else:
        st.subheader("Run Comparison")
        st.dataframe(
            runs_df.style.highlight_max(subset=["val_accuracy"], color="#c9f2c9"),
            use_container_width=True
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Validation Accuracy by Run")
            st.bar_chart(runs_df.set_index("run_name")["val_accuracy"])
        with col2:
            st.subheader("Train Loss by Run")
            st.bar_chart(runs_df.set_index("run_name")["train_loss"])

        best_row = runs_df.loc[runs_df["val_accuracy"].idxmax()]
        st.success(
            f"Best run: **{best_row['run_name']}** "
            f"(lr={best_row['lr']}, epochs={best_row['epochs']}, "
            f"batch_size={best_row['batch_size']}) → "
            f"val_accuracy = {best_row['val_accuracy']:.4f}"
        )
        st.caption("This is the model currently served in the Live Inference tab.")

with tab2:
    st.subheader("Try the Best Model")
    uploaded = st.file_uploader("Upload an image (CIFAR-10 style object photo)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", width=200)

        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs))

        st.metric("Prediction", CLASSES[pred_idx], f"{probs[pred_idx]*100:.1f}% confidence")

        prob_df = pd.DataFrame({"class": CLASSES, "probability": probs.numpy()})
        st.bar_chart(prob_df.set_index("class"))
    else:
        st.info("Upload an image to run inference with the current best model.")

st.divider()
st.caption(
    "Pipeline: train.py logs experiments to MLflow → best model + run summary exported → "
    "GitHub Actions validates the pipeline on every push → dashboard deployed on Streamlit Cloud."
)
