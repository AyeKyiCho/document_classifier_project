import streamlit as st
import torch
import time
import re
import json
import os
import pdfplumber

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Document Classifier",
    page_icon="📄",
    layout="centered"
)

# -----------------------------
# Load Custom CSS
# -----------------------------
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("""
            <style>
            .main { padding: 2rem; }
            .stButton > button {
                width: 100%;
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            .upload-box {
                border: 2px dashed #4CAF50;
                border-radius: 10px;
                padding: 2rem;
                text-align: center;
                background-color: #f9f9f9;
            }
            .result-box {
                padding: 1.5rem;
                border-radius: 10px;
                margin-top: 1rem;
            }
            .result-success {
                background-color: #e8f5e9;
                border-left: 5px solid #4CAF50;
            }
            .result-warning {
                background-color: #fff3e0;
                border-left: 5px solid #FF9800;
            }
            .result-other {
                background-color: #f5f5f5;
                border-left: 5px solid #9E9E9E;
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
            .debug-box {
                background-color: #fff3cd;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
                border: 1px solid #ffc107;
            }
            </style>
        """, unsafe_allow_html=True)
        st.warning("⚠️ styles.css not found. Using fallback styling.")

load_css()

# -----------------------------
# Load Categories from JSON
# -----------------------------
@st.cache_resource
def load_categories():
    json_path = os.path.join(os.path.dirname(__file__), "categories.json")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            categories = json.load(f)
        
        # REMOVE CODE CATEGORY if it exists
        if "Code" in categories:
            del categories["Code"]
            st.info("ℹ️ 'Code' category removed from classification")
        
        return categories
    except Exception as e:
        st.error(f"Error loading categories.json: {e}")
        return {}

CATEGORIES = load_categories()

# -----------------------------
# Load Pretrained NLP Model
# -----------------------------
@st.cache_resource
def load_model():
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

if not CATEGORIES:
    st.error("No categories loaded. Please check categories.json file.")
    st.stop()

# -----------------------------
# Classification Logic
# -----------------------------
def classify_document(text, slider_threshold, debug=False):
    text_lower = text.lower()
    text_words = set(text_lower.split())
    
    scores = {}
    matched_keywords = {}
    unique_matches = {}
    all_matches = {}
    
    # Keyword scoring
    for category, info in CATEGORIES.items():
        score = 0
        matches = []
        unique_count = 0
        category_matches = {}
        
        for keyword in info["keywords"]:
            count = text_lower.count(keyword)
            if count > 0:
                score += count
                matches.append(keyword)
                unique_count += 1
                category_matches[keyword] = count
        
        unique_matches[category] = unique_count
        
        if len(matches) > 2:
            score *= 1.5
        
        scores[category] = score
        matched_keywords[category] = matches
        all_matches[category] = category_matches
    
    # Best category
    predicted_label = max(scores, key=scores.get)
    max_score = scores[predicted_label]
    max_unique = unique_matches[predicted_label]
    
    threshold = slider_threshold
    
    # Determine "Other"
    is_other = False
    other_reason = ""
    
    word_count = len(text_words)
    
    if max_unique < threshold:
        is_other = True
        other_reason = f"Only {max_unique} unique keyword matches found (need {threshold})"
    
    if word_count > 20 and max_score / word_count < 0.05:
        is_other = True
        other_reason = f"Very few keyword matches ({max_score} matches in {word_count} words)"
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) >= 2:
        if sorted_scores[1][1] / sorted_scores[0][1] > 0.7:
            is_other = True
            other_reason = f"Ambiguous - similar scores: {sorted_scores[0][0]} vs {sorted_scores[1][0]}"
    
    if word_count < 10:
        is_other = True
        other_reason = "Document too short"
    
    total_scores = sum(scores.values())
    confidence = max_score / total_scores if total_scores > 0 else 0
    
    # Sentiment fallback
    if max_score == 0:
        try:
            inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits).item()
                probabilities = torch.softmax(logits, dim=1)
                confidence = float(probabilities[0][predicted_class])
                confidence = min(confidence, 0.4)
            
            predicted_label = "Invoice" if predicted_class == 1 else "Email"
            
            if confidence < 0.3:
                is_other = True
                other_reason = "Low confidence sentiment fallback"
        
        except Exception as e:
            is_other = True
            other_reason = f"Sentiment error: {str(e)[:50]}"
    
    if is_other:
        predicted_label = "Other / Unknown"
        confidence = max(confidence, 0.1)
    
    # Add icon
    if predicted_label in CATEGORIES:
        icon = CATEGORIES[predicted_label].get("icon", "")
        if icon:
            predicted_label = f"{icon} {predicted_label}"
    else:
        predicted_label = "❓ Other / Unknown"
    
    if debug:
        return predicted_label, confidence, scores, matched_keywords, is_other, other_reason, all_matches
    
    return predicted_label, confidence, scores, matched_keywords, is_other, other_reason

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Intelligent Document Classifier")
st.markdown("### Upload a text or PDF file and get instant AI classification")

# Sidebar with settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    slider_threshold = st.slider(
        "Minimum keyword matches required",
        min_value=1,
        max_value=10,
        value=2,
        help="Higher values make classification more strict"
    )
    
    debug_mode = st.checkbox(
        "🐞 Debug Mode",
        value=False,
        help="Show detailed keyword matching information"
    )
    
    st.divider()
    st.header("Supported Types")
    for category, info in CATEGORIES.items():
        st.write(f"{info.get('icon', '📄')} {category}")
    
    st.divider()
    with st.expander("ℹ️ How it works"):
        st.write("""
        1. Upload a .txt or .pdf file  
        2. The app analyzes keywords in your document  
        3. Each category gets a score based on keyword matches  
        4. The category with the highest score is selected  
        5. If the score is too low, it's marked as 'Other'  
        """)
    
    st.divider()
    st.caption("Version 2.4 • Code Category Removed")

# -----------------------------
# File Upload (TXT + PDF)
# -----------------------------
uploaded_file = st.file_uploader(
    "**Drop your .txt or .pdf file here**",
    type=["txt", "pdf"],
    help="Supports .txt and .pdf files up to 5MB"
)

# -----------------------------
# File Processing
# -----------------------------
if uploaded_file:
    with st.spinner("Processing document..."):
        
        # TXT
        if uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        
        # PDF
        elif uploaded_file.type == "application/pdf":
            text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            
            if not text.strip():
                st.error("❌ Could not extract text from this PDF. It may be scanned.")
                st.stop()
        
        else:
            st.error("Unsupported file type.")
            st.stop()
        
        time.sleep(0.3)
    
    # File Info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📁 File Name", uploaded_file.name)
    with col2:
        st.metric("📏 Size", f"{len(text):,} chars")
    with col3:
        st.metric("📝 Words", len(text.split()))
    
    # Preview
    st.subheader("📖 Document Preview")
    preview_text = text[:500] + ("..." if len(text) > 500 else "")
    st.markdown(f'<div class="preview-box">{preview_text}</div>', unsafe_allow_html=True)
    
    # Classification Button
    if st.button("🔍 Classify Document", use_container_width=True):
        if debug_mode:
            predicted_label, confidence, scores, matched_keywords, is_other, other_reason, all_matches = classify_document(
                text, slider_threshold, debug=True
            )
        else:
            predicted_label, confidence, scores, matched_keywords, is_other, other_reason = classify_document(
                text, slider_threshold
            )
        
        st.subheader("Classification Result")
        
        box_class = "result-other" if is_other else ("result-success" if confidence > 0.7 else "result-warning")
        
        st.markdown(f'''
            <div class="result-box {box_class}">
                <h2>{predicted_label}</h2>
                <p>Confidence: {confidence:.1%}</p>
                {f"<p>💡 {other_reason}</p>" if is_other else ""}
            </div>
        ''', unsafe_allow_html=True)
        
        st.progress(min(confidence, 1.0))
        
        # Detailed Analysis
        with st.expander("📊 Detailed Analysis", expanded=True):
            st.write("**Category Scores:**")
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            max_score = max(scores.values()) if scores.values() else 1
            for category, score in sorted_scores:
                bar_width = score / max_score if max_score > 0 else 0
                icon = CATEGORIES.get(category, {}).get("icon", "📄")
                st.write(f"{icon} **{category}**: {score} matches")
                st.progress(bar_width)
            
            st.write("**Keywords Found:**")
            for category, keywords in matched_keywords.items():
                if keywords:
                    st.write(f"**{category}:** {', '.join(keywords)}")
            
            # Debug Information
            if debug_mode:
                st.subheader("🐞 Debug Information")
                
                st.write("**Detailed Keyword Matches:**")
                for category, matches in all_matches.items():
                    if matches:
                        st.write(f"**{category}:**")
                        for keyword, count in matches.items():
                            st.write(f"  - '{keyword}': {count} time(s)")
                
                st.write("**Text Sample (first 500 chars):**")
                st.code(text[:500], language='text')
                
                from collections import Counter
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                word_freq = Counter(words).most_common(20)
                st.write("**Most Common Words:**")
                for word, count in word_freq:
                    st.write(f"  - '{word}': {count}")

else:
    st.info("👆 Upload a .txt or .pdf file to begin classification")

st.divider()
st.caption("Built with Transformers • Powered by DistilBERT")