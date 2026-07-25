import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# -----------------------------
# Load Pretrained NLP Model
# -----------------------------
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Custom labels for your classifier
LABELS = ["invoice", "resume", "report", "email", "contract"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Intelligent Document Classifier (NLP)")
st.write("Upload a text file and the AI will classify it into a category.")

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file:
    # Read text
    text = uploaded_file.read().decode("utf-8")

    st.subheader("Document Preview")
    st.write(text[:500] + ("..." if len(text) > 500 else ""))

    # Encode text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()

    # Map prediction to your custom labels
    predicted_label = LABELS[predicted_class % len(LABELS)]

    st.subheader("Predicted Category")
    st.success(predicted_label)
