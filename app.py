import streamlit as st
import torch
import time
import re
import json
import os

# Import only the specific transformers components you need
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
    """Load custom CSS from file"""
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback to minimal styling if CSS file not found
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
            </style>
        """, unsafe_allow_html=True)
        st.warning("⚠️ styles.css not found. Using fallback styling.")

# Load CSS
load_css()

# -----------------------------
# Load Categories from JSON
# -----------------------------
@st.cache_resource
def load_categories():
    """Load categories from JSON file"""
    json_path = os.path.join(os.path.dirname(__file__), "categories.json")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            categories = json.load(f)
        return categories
    except FileNotFoundError:
        st.error(f"Categories file not found at: {json_path}")
        st.info("Please create a 'categories.json' file in the same directory as app.py")
        return {}
    except json.JSONDecodeError as e:
        st.error(f"Error parsing categories.json: {e}")
        return {}

# Load categories
CATEGORIES = load_categories()

# -----------------------------
# Load Pretrained NLP Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Stop if no categories loaded
if not CATEGORIES:
    st.error("No categories loaded. Please check categories.json file.")
    st.stop()

def classify_document(text):
    """Classify document using keyword matching with threshold and fallback"""
    text_lower = text.lower()
    text_words = set(text_lower.split())
    
    # Detect if it's code
    code_patterns = re.findall(r'\b(import|def|class|function|return|print|if|else|for|while|try|except)\b', text_lower)
    has_code = len(code_patterns) > 5
    
    scores = {}
    matched_keywords = {}
    unique_matches = {}
    
    # Calculate scores for each category
    for category, info in CATEGORIES.items():
        score = 0
        matches = []
        unique_count = 0
        
        for keyword in info["keywords"]:
            count = text_lower.count(keyword)
            if count > 0:
                score += count
                matches.append(keyword)
                unique_count += 1
        
        unique_matches[category] = unique_count
        
        if len(matches) > 2:
            score *= 1.5
        
        scores[category] = score
        matched_keywords[category] = matches
    
    # Code detection override
    if has_code and "Code" in CATEGORIES and scores.get("Code", 0) > 0:
        scores["Code"] *= 2
    
    # Get best category
    predicted_label = max(scores, key=scores.get)
    max_score = scores[predicted_label]
    max_unique = unique_matches[predicted_label]
    
    # Get threshold for best category
    threshold = CATEGORIES.get(predicted_label, {}).get("threshold", 3)
    
    # Check if document is "Other"
    is_other = False
    other_reason = ""
    
    # Condition 1: Not enough unique keyword matches
    if max_unique < threshold:
        is_other = True
        other_reason = f"Only {max_unique} unique keyword matches found (need {threshold})"
    
    # Condition 2: Total score too low relative to document length
    word_count = len(text_words)
    if word_count > 20 and max_score / word_count < 0.05:
        is_other = True
        other_reason = f"Very few keyword matches ({max_score} matches in {word_count} words)"
    
    # Condition 3: Multiple categories have similar scores
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) >= 2:
        top_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1]
        if top_score > 0 and second_score > 0 and (second_score / top_score) > 0.7:
            is_other = True
            other_reason = f"Ambiguous - similar scores: {sorted_scores[0][0]} ({top_score}) vs {sorted_scores[1][0]} ({second_score})"
    
    # Condition 4: Document is too short
    if word_count < 10:
        is_other = True
        other_reason = "Document too short for reliable classification"
    
    # Calculate confidence
    total_scores = sum(scores.values())
    confidence = (max_score / total_scores) if total_scores > 0 else 0
    
    # If no keywords found, use sentiment analysis
    if max_score == 0:
        try:
            inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits).item()
                probabilities = torch.softmax(logits, dim=1)
                confidence = probabilities[0][predicted_class].item()
                confidence = min(confidence, 0.4)
            
            if predicted_class == 1:
                predicted_label = "Invoice"
            else:
                predicted_label = "Email"
            
            if confidence < 0.3:
                is_other = True
                other_reason = "Low confidence sentiment analysis fallback"
        except Exception as e:
            is_other = True
            other_reason = f"Error in sentiment analysis: {str(e)[:50]}"
    
    if is_other:
        predicted_label = "Other / Unknown"
        confidence = max(confidence, 0.1)
    
    # Add icon to predicted label if it exists in categories
    if predicted_label in CATEGORIES:
        icon = CATEGORIES[predicted_label].get("icon", "")
        if icon:
            predicted_label = f"{icon} {predicted_label}"
    elif predicted_label == "Other / Unknown":
        predicted_label = "❓ Other / Unknown"
    
    return predicted_label, confidence, scores, matched_keywords, is_other, other_reason

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Intelligent Document Classifier")
st.markdown("### Upload a text file and get instant AI classification")

# Sidebar with settings
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Minimum keyword matches required",
        min_value=1,
        max_value=10,
        value=3,
        help="Higher values make classification more strict"
    )
    st.caption("Documents below threshold will be marked as 'Other'")
    
    st.divider()
    st.header("Supported Types")
    # Display categories from JSON
    for category, info in CATEGORIES.items():
        st.write(f"{info.get('icon', '📄')} {category}")
    
    st.divider()
    with st.expander("ℹ️ How it works"):
        st.write("""
        1. Upload a .txt file
        2. The app analyzes keywords in your document
        3. Each category gets a score based on keyword matches
        4. The category with the highest score is selected
        5. If the score is too low, it's marked as 'Other'
        """)
    
    st.divider()
    st.caption("Version 2.0")

# Create upload section
with st.container():
   
    uploaded_file = st.file_uploader(
        "**Drop your .txt file here** or click to browse",
        type=["txt"],
        help="Supports .txt files up to 5MB"
    )
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    with st.spinner("Processing document..."):
        text = uploaded_file.read().decode("utf-8")
        time.sleep(0.3)
    
    # File info with improved styling
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("📁 File Name", uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("📏 Size", f"{len(text):,} chars")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("📝 Words", len(text.split()))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Document preview
    st.subheader("📖 Document Preview")
    preview_text = text[:500] + ("..." if len(text) > 500 else "")
    st.markdown(f'<div class="preview-box">{preview_text}</div>', unsafe_allow_html=True)
    
    # Classification button
    if st.button("🔍 Classify Document", use_container_width=True):
        with st.spinner("AI is analyzing..."):
            predicted_label, confidence, scores, matched_keywords, is_other, other_reason = classify_document(text)
        
        # Display result with appropriate styling
        st.subheader("Classification Result")
        
        if is_other:
            box_class = "result-other"           
        elif confidence > 0.7:
            box_class = "result-success"           
        else:
            box_class = "result-warning"
            
        st.markdown(f'''
            <div class="result-box {box_class}">
                <h2 style="margin:0;">{predicted_label}</h2>
                <p style="margin:0.5rem 0 0 0; color:#555;">
                    Confidence: {confidence:.1%}
                </p>
                {f'<p style="margin:0.3rem 0 0 0; color:#888; font-size:0.9rem;">💡 {other_reason}</p>' if is_other else ''}
            </div>
        ''', unsafe_allow_html=True)
        
        # Confidence meter
        st.progress(min(confidence, 1.0))
        
        # Detailed analysis
        with st.expander("📊 Detailed Analysis", expanded=True):
            st.write("**Category Scores:**")
            
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            for category, score in sorted_scores:
                if score > 0:
                    max_score = max(scores.values()) if max(scores.values()) > 0 else 1
                    bar_width = min(score / max_score * 100, 100)
                    icon = CATEGORIES.get(category, {}).get("icon", "📄")
                    st.write(f"{icon} **{category}**: {score} matches")
                    st.progress(bar_width / 100)
            
            if is_other:
                st.warning(f"⚠️ **Document classified as 'Other'**")
                st.info(f"Reason: {other_reason}")
            
            st.write("**Keywords Found:**")
            found_any = False
            for category, keywords in matched_keywords.items():
                if keywords:
                    found_any = True
                    # Show keywords with chips
                    keywords_html = ' '.join([f'<span class="category-chip">{k}</span>' for k in keywords[:10]])
                    st.markdown(f"**{category}:** {keywords_html}", unsafe_allow_html=True)
            
            if not found_any:
                st.write("No keywords matched")
            
            code_patterns = re.findall(r'\b(import|def|class|function|return|print|if|else|for|while|try|except)\b', text.lower())
            if code_patterns:
                st.info(f"💻 **Code detected!** Found these patterns: {', '.join(set(code_patterns[:10]))}")
            
            st.write("**Document Statistics:**")
            words = text.split()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Words", len(words))
            with col2:
                st.metric("Unique Words", len(set(words)))
            with col3:
                st.metric("Avg Word Length", f"{sum(len(w) for w in words) / len(words) if words else 0:.1f}")
else:
    st.info("👆 Upload a .txt file to begin classification")
    st.caption("Supported categories: Email, Invoice, Report, Resume, Contract, Code")
    st.caption("Documents that don't match any category will be marked as 'Other'")

st.divider()
st.caption("Built with Transformers • Powered by DistilBERT")