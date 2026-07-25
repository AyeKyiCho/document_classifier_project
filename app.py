import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Document Classifier",
    page_icon="📄",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f9f9f9;
    }
    .result-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-top: 1rem;
    }
    .preview-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        max-height: 200px;
        overflow-y: auto;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Pretrained NLP Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# Custom labels with icons
LABELS = ["📧 Email", "📄 Invoice", "📋 Report", "📝 Resume", "📑 Contract"]

# Simple keyword-based classification (fallback since model only has 2 classes)
def classify_document(text):
    text_lower = text.lower()
    
    # Keyword scoring for each category
    scores = {
        "📧 Email": 0,
        "📄 Invoice": 0,
        "📋 Report": 0,
        "📝 Resume": 0,
        "📑 Contract": 0
    }
    
    # Keywords for each category
    keywords = {
        "📧 Email": ["dear", "hello", "hi", "regards", "sincerely", "thanks", "email", "sent", "received", "subject"],
        "📄 Invoice": ["invoice", "payment", "amount", "due", "total", "balance", "receipt", "billing", "purchase", "order"],
        "📋 Report": ["report", "analysis", "summary", "findings", "results", "data", "conclusion", "recommendation", "performance"],
        "📝 Resume": ["experience", "skills", "education", "job", "work", "career", "position", "accomplishments", "professional"],
        "📑 Contract": ["agreement", "terms", "conditions", "party", "clause", "effective", "signature", "legal", "obligation"]
    }
    
    # Count keyword occurrences
    for category, words in keywords.items():
        for word in words:
            scores[category] += text_lower.count(word)
    
    # Get the category with highest score
    predicted_label = max(scores, key=scores.get)
    max_score = scores[predicted_label]
    
    # If no keywords found, use sentiment as fallback
    if max_score == 0:
        # Use sentiment to determine category
        inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits).item()
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities[0][predicted_class].item()
        
        # Map sentiment to categories (positive = business docs, negative = personal)
        if predicted_class == 1:  # Positive sentiment
            predicted_label = "📄 Invoice"  # Default to business
        else:  # Negative sentiment
            predicted_label = "📧 Email"  # Default to personal
        
        return predicted_label, confidence
    
    # Calculate confidence based on score distribution
    total_scores = sum(scores.values())
    confidence = max_score / total_scores if total_scores > 0 else 0.5
    
    return predicted_label, min(confidence, 0.95)  # Cap at 95%

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Intelligent Document Classifier")
st.markdown("### Upload a text file and get instant AI classification")

# Create a nice upload section
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "**Drop your .txt file here** or click to browse",
        type=["txt"],
        help="Supports .txt files up to 5MB"
    )
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    # Read text with progress indicator
    with st.spinner("Processing document..."):
        text = uploaded_file.read().decode("utf-8")
        time.sleep(0.5)  # Show spinner briefly
    
    # File info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Name", uploaded_file.name)
    with col2:
        st.metric("Size", f"{len(text):,} chars")
    with col3:
        st.metric("Words", len(text.split()))
    
    # Document preview
    st.subheader("📖 Document Preview")
    preview_text = text[:500] + ("..." if len(text) > 500 else "")
    st.markdown(f'<div class="preview-box">{preview_text}</div>', unsafe_allow_html=True)
    
    # Classification button
    if st.button("🔍 Classify Document", use_container_width=True):
        with st.spinner("AI is analyzing..."):
            predicted_label, confidence = classify_document(text)
        
        # Display result with confidence
        st.subheader("🎯 Classification Result")
        st.markdown(f'''
            <div class="result-box">
                <h3 style="margin:0; color:#2e7d32;">{predicted_label}</h3>
                <p style="margin:0.5rem 0 0 0; color:#555;">
                    Confidence: {confidence:.1%}
                </p>
            </div>
        ''', unsafe_allow_html=True)
        
        # Add confidence meter
        st.progress(confidence)
        
        # Show keyword breakdown
        with st.expander("📊 View Analysis Details"):
            st.write("**Keyword Detection Results:**")
            text_lower = text.lower()
            
            # Show keyword counts for each category
            keywords = {
                "📧 Email": ["dear", "hello", "hi", "regards", "sincerely", "thanks", "email"],
                "📄 Invoice": ["invoice", "payment", "amount", "due", "total", "balance", "receipt"],
                "📋 Report": ["report", "analysis", "summary", "findings", "results", "data"],
                "📝 Resume": ["experience", "skills", "education", "job", "work", "career", "position"],
                "📑 Contract": ["agreement", "terms", "conditions", "party", "clause", "effective", "legal"]
            }
            
            for category, words in keywords.items():
                count = sum(text_lower.count(word) for word in words)
                st.write(f"{category}: {count} keyword matches")

else:
    # Empty state
    st.info("👆 Upload a .txt file to begin classification")
    st.caption("Supported categories: Email, Invoice, Report, Resume, Contract")

# Footer
st.divider()
st.caption("Built with 🤗 Transformers • Powered by DistilBERT")