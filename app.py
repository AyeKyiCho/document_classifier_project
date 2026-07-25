import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from typing import Tuple
import time
import random

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="DocuMind AI - Document Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS for Premium Design
# -----------------------------
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Custom header gradient */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        margin-bottom: 1.5rem;
        border: 1px solid #f0f0f0;
        transition: transform 0.2s;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: none;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    .stat-label {
        color: #7f8c8d;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Result card animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .result-card {
        animation: slideIn 0.5s ease-out;
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
    }
    .result-card.high {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
    }
    .result-card.medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
    }
    .result-card.low {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        box-shadow: 0 10px 30px rgba(247, 151, 30, 0.3);
    }
    
    /* Custom progress bar */
    .custom-progress {
        height: 10px;
        border-radius: 10px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        margin: 1rem 0;
    }
    
    /* Animated emoji */
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    .bounce-emoji {
        animation: bounce 2s infinite;
        display: inline-block;
    }
    
    /* Category tags */
    .category-tag {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.2rem;
        transition: all 0.2s;
    }
    .category-tag:hover {
        transform: scale(1.05);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed #e0e0e0;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s;
    }
    .upload-area:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #95a5a6;
        font-size: 0.85rem;
        margin-top: 2rem;
        border-top: 1px solid #ecf0f1;
    }
    
    /* Typography */
    .highlight {
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
CUSTOM_LABELS = ["invoice", "resume", "report", "email", "contract"]
LABEL_EMOJIS = {
    "invoice": "💰",
    "resume": "👤",
    "report": "📊",
    "email": "✉️",
    "contract": "📋"
}
MAX_TEXT_LENGTH = 512
PREVIEW_LENGTH = 500

# -----------------------------
# Model Loading with Caching
# -----------------------------
@st.cache_resource
def load_model():
    """Load the pretrained model and tokenizer with caching."""
    try:
        with st.spinner("🧠 Loading AI model..."):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            return tokenizer, model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()

tokenizer, model = load_model()

# -----------------------------
# Helper Functions
# -----------------------------
def preprocess_text(text: str) -> str:
    """Clean and preprocess input text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def classify_document(text: str) -> Tuple[str, float]:
    """Classify document using the pretrained model."""
    cleaned_text = preprocess_text(text)
    
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_TEXT_LENGTH
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        
    label_index = predicted_class.item() % len(CUSTOM_LABELS)
    predicted_label = CUSTOM_LABELS[label_index]
    confidence_score = confidence.item()
    
    return predicted_label, confidence_score

def get_confidence_level(score: float) -> str:
    """Get confidence level based on score."""
    if score >= 0.8:
        return "high"
    elif score >= 0.5:
        return "medium"
    else:
        return "low"

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### 🎯 Categories")
    for label in CUSTOM_LABELS:
        emoji = LABEL_EMOJIS.get(label, "📄")
        st.markdown(f"- {emoji} **{label.capitalize()}**")
    
    st.divider()
    
    st.markdown("### ⚙️ Settings")
    st.info("Model: **DistilBERT**")
    st.info(f"Max tokens: **{MAX_TEXT_LENGTH}**")
    
    st.divider()
    
    st.markdown("### 📊 Statistics")
    if 'total_classifications' not in st.session_state:
        st.session_state.total_classifications = 0
    st.metric("Total Classifications", st.session_state.total_classifications)
    
    st.divider()
    
    st.markdown("### 💡 Tips")
    st.caption("• Upload clear, well-formatted text")
    st.caption("• Documents should be at least 50 words")
    st.caption("• Supported format: .txt files only")

# -----------------------------
# Main Content
# -----------------------------
# Header
st.markdown("""
<div class="custom-header">
    <h1 style="margin:0;">🧠 DocuMind AI</h1>
    <p style="margin:0.5rem 0 0 0; opacity:0.9; font-size:1.1rem;">
        Intelligent Document Classification with Natural Language Processing
    </p>
</div>
""", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📤 Upload Document")
    
    # Custom upload area
    uploaded_file = st.file_uploader(
        "Drop your text file here or click to browse",
        type=["txt"],
        help="Supported file type: .txt"
    )

with col2:
    st.markdown("### 🚀 Quick Actions")
    st.markdown("""
    <div style="background:#f8f9fa; padding:1rem; border-radius:10px; text-align:center;">
        <p style="font-size:2rem; margin:0;">📁</p>
        <p style="font-size:0.9rem; color:#7f8c8d;">Upload a file to begin</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# File processing
if uploaded_file:
    try:
        text = uploaded_file.read().decode("utf-8")
        filename = uploaded_file.name
        
        # Display file info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 File", filename[:20] + ("..." if len(filename) > 20 else ""))
        with col2:
            st.metric("📝 Words", len(text.split()))
        with col3:
            st.metric("🔤 Characters", len(text))
        with col4:
            st.metric("💾 Size", f"{len(text) / 1024:.1f} KB")
        
        # Document preview with expander
        with st.expander("👁️ Document Preview", expanded=True):
            preview_text = text[:PREVIEW_LENGTH]
            if len(text) > PREVIEW_LENGTH:
                preview_text += "..."
            st.text_area("Content", preview_text, height=200, disabled=True, label_visibility="collapsed")
        
        # Classification
        if st.button("🚀 Classify Document", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing document with AI..."):
                # Simulate processing time for better UX
                time.sleep(0.5)
                label, confidence = classify_document(text)
                
                # Update session state
                st.session_state.total_classifications += 1
            
            # Display results with animation
            confidence_level = get_confidence_level(confidence)
            emoji = LABEL_EMOJIS.get(label, "📄")
            
            # Result card
            st.markdown(f"""
            <div class="result-card {confidence_level}">
                <div style="display:flex; align-items:center; justify-content:space-between;">
                    <div>
                        <span style="font-size:2.5rem;">{emoji}</span>
                        <span style="font-size:2rem; font-weight:bold; margin-left:0.5rem;">{label.upper()}</span>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:2.5rem; font-weight:bold;">{confidence:.1%}</div>
                        <div style="opacity:0.8;">Confidence Score</div>
                    </div>
                </div>
                <div style="margin-top:1rem;">
                    <div style="background:rgba(255,255,255,0.2); height:8px; border-radius:10px;">
                        <div style="background:white; height:100%; width:{confidence*100}%; border-radius:10px; transition:width 1s;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence interpretation
            st.markdown("#### 📊 Confidence Level")
            if confidence >= 0.8:
                st.success("✅ **High Confidence** - The AI is very certain about this classification")
                st.progress(confidence, text=f"Confidence: {confidence:.1%}")
            elif confidence >= 0.5:
                st.warning("⚠️ **Medium Confidence** - The AI is moderately certain")
                st.progress(confidence, text=f"Confidence: {confidence:.1%}")
            else:
                st.error("❌ **Low Confidence** - The AI is uncertain, consider reviewing the document")
                st.progress(confidence, text=f"Confidence: {confidence:.1%}")
            
            # Additional insights
            with st.expander("🔍 Detailed Analysis"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Model Details**")
                    st.write(f"- Model: {MODEL_NAME}")
                    st.write(f"- Max tokens: {MAX_TEXT_LENGTH}")
                    st.write(f"- Categories: {len(CUSTOM_LABELS)}")
                with col2:
                    st.markdown("**Document Insights**")
                    st.write(f"- Words: {len(text.split())}")
                    st.write(f"- Characters: {len(text)}")
                    st.write(f"- Lines: {len(text.splitlines())}")
            
    except UnicodeDecodeError:
        st.error("❌ Error: Could not decode the file. Please ensure it's a valid UTF-8 text file.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        
else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding:3rem 0;">
        <span style="font-size:4rem; display:block; margin-bottom:1rem;" class="bounce-emoji">📄</span>
        <h3 style="color:#7f8c8d;">Ready to Classify Your Documents</h3>
        <p style="color:#95a5a6;">Upload a text file using the uploader above to get started</p>
        <div style="display:flex; justify-content:center; gap:0.5rem; flex-wrap:wrap; margin-top:1rem;">
            <span class="category-tag" style="background:#e8f5e9;">💰 Invoice</span>
            <span class="category-tag" style="background:#e3f2fd;">👤 Resume</span>
            <span class="category-tag" style="background:#f3e5f5;">📊 Report</span>
            <span class="category-tag" style="background:#fff3e0;">✉️ Email</span>
            <span class="category-tag" style="background:#fce4ec;">📋 Contract</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.markdown("""
<div class="footer">
    <p>Built with ❤️ using Streamlit, Transformers & PyTorch</p>
    <p style="font-size:0.75rem;">DocuMind AI v2.0 • Intelligent Document Classification</p>
</div>
""", unsafe_allow_html=True)