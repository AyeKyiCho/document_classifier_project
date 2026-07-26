import streamlit as st
import torch
import time
import re
import json
import os
import pdfplumber
from datetime import datetime
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Document Classifier",
    page_icon="📄",
    layout="wide",  # Better use of space
    initial_sidebar_state="expanded"
)

# -----------------------------
# Session State
# -----------------------------
if 'history' not in st.session_state:
    st.session_state.history = []
if 'classification_count' not in st.session_state:
    st.session_state.classification_count = 0
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# -----------------------------
# Load Custom CSS
# -----------------------------
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
        return True
    except FileNotFoundError:
        st.markdown("""
            <style>
            .main { padding: 2rem; }
            .stButton > button {
                width: 100%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-weight: bold;
                border: none;
                transition: all 0.3s ease;
            }
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            .upload-box {
                border: 2px dashed #667eea;
                border-radius: 12px;
                padding: 2.5rem;
                text-align: center;
                background: linear-gradient(135deg, #f8f9ff 0%, #f0f1ff 100%);
                transition: all 0.3s ease;
            }
            .upload-box:hover {
                border-color: #764ba2;
                transform: scale(1.01);
            }
            .result-box {
                padding: 1.5rem;
                border-radius: 12px;
                margin-top: 1rem;
                animation: fadeIn 0.5s ease;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .result-success {
                background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
                border-left: 5px solid #4CAF50;
            }
            .result-warning {
                background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
                border-left: 5px solid #FF9800;
            }
            .result-other {
                background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
                border-left: 5px solid #9E9E9E;
            }
            .preview-box {
                background-color: #f5f5f5;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
                max-height: 200px;
                overflow-y: auto;
                font-family: 'Courier New', monospace;
                border: 1px solid #e0e0e0;
                line-height: 1.6;
            }
            .metric-card {
                background: white;
                border-radius: 8px;
                padding: 1rem;
                text-align: center;
                border: 1px solid #e0e0e0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .category-chip {
                display: inline-block;
                padding: 0.3rem 0.8rem;
                margin: 0.2rem;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 500;
                background: #e3f2fd;
                color: #1976d2;
                border: 1px solid #bbdefb;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            </style>
        """, unsafe_allow_html=True)
        return False

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
        
        # Remove Code category if exists
        if "Code" in categories:
            del categories["Code"]
        
        return categories
    except Exception as e:
        st.error(f"❌ Error loading categories.json: {e}")
        return {}

CATEGORIES = load_categories()

# -----------------------------
# Load Pretrained NLP Model
# -----------------------------
@st.cache_resource
def load_model():
    try:
        MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        return tokenizer, model, True
    except Exception as e:
        st.warning(f"⚠️ Model not available: {str(e)[:50]}")
        return None, None, False

tokenizer, model, model_loaded = load_model()

if not CATEGORIES:
    st.error("No categories loaded. Please check categories.json file.")
    st.stop()

# -----------------------------
# Helper Functions
# -----------------------------
def extract_top_keywords(text, n=10):
    """Extract most frequent keywords from text"""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has', 'his', 'have', 'from', 'with', 'this', 'that', 'they', 'will'}
    words = [w for w in words if w not in stopwords]
    return Counter(words).most_common(n)

def get_document_summary(text, max_sentences=2):
    """Get document summary"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return ' '.join(sentences[:max_sentences])

def export_results(uploaded_file, predicted_label, confidence, scores, text):
    """Export results as CSV"""
    import pandas as pd
    results = {
        'Filename': uploaded_file.name,
        'Category': predicted_label,
        'Confidence': f"{confidence:.1%}",
        'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Word Count': len(text.split()),
        'Character Count': len(text),
    }
    for category, score in scores.items():
        results[f'Score_{category}'] = score
    return pd.DataFrame([results])

# -----------------------------
# Classification Logic
# -----------------------------
def classify_document(text, slider_threshold, debug=False):
    text_lower = text.lower()
    text_words = set(text_lower.split())

    # Detect code
    code_patterns = re.findall(
        r'\b(import|def|class|function|return|print|if|else|for|while|try|except)\b',
        text_lower
    )
    has_code = len(code_patterns) > 5

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
                weight = info.get("weight", 1.0)
                score += count * weight
                matches.append(keyword)
                unique_count += 1
                category_matches[keyword] = count

        unique_matches[category] = unique_count

        if len(matches) > 2:
            score *= 1.5

        scores[category] = score
        matched_keywords[category] = matches
        all_matches[category] = category_matches

    # Code override
    if has_code and "Code" in CATEGORIES and scores.get("Code", 0) > 0:
        scores["Code"] *= 2

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
        other_reason = f"Only {max_unique} unique keywords (need {threshold})"

    if word_count > 20 and max_score / word_count < 0.05:
        is_other = True
        other_reason = f"Low keyword density ({max_score}/{word_count})"

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) >= 2:
        if sorted_scores[1][1] / sorted_scores[0][1] > 0.7:
            is_other = True
            other_reason = f"Ambiguous: {sorted_scores[0][0]} vs {sorted_scores[1][0]}"

    if word_count < 10:
        is_other = True
        other_reason = "Document too short"

    total_scores = sum(scores.values())
    confidence = max_score / total_scores if total_scores > 0 else 0

    # Sentiment fallback
    if max_score == 0 and tokenizer and model and model_loaded:
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
                other_reason = "Low confidence fallback"

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

# Model status
if model_loaded:
    st.success("✅ AI Model: Ready")
else:
    st.info("ℹ️ Running in keyword-matching mode (faster)")

# Sidebar with settings
with st.sidebar:
    st.header("⚙️ Settings")

    slider_threshold = st.slider(
        "Minimum keyword matches required",
        min_value=1,
        max_value=10,
        value=3,
        help="Higher values make classification more strict"
    )

    debug_mode = st.checkbox(
        "🐞 Debug Mode",
        value=st.session_state.debug_mode,
        help="Show detailed keyword matching information"
    )
    st.session_state.debug_mode = debug_mode

    st.divider()
    st.header("📌 Supported Types")
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
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.classification_count = 0
        st.success("History cleared!")

    st.divider()
    st.caption(f"Version 2.5 • {datetime.now().year}")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "**Drop your .txt or .pdf file here** or click to browse",
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
            text = uploaded_file.read().decode("utf-8", errors="ignore")

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

    # -----------------------------
    # File Info - Enhanced Metrics
    # -----------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📁 File Name", uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📏 Size", f"{len(text):,} chars")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📝 Words", len(text.split()))
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        has_code = bool(re.search(r'\b(import|def|class|function|return)\b', text))
        st.metric("💻 Code", "Yes" if has_code else "No")
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # Preview
    # -----------------------------
    st.subheader("📖 Document Preview")
    preview_text = text[:500] + ("..." if len(text) > 500 else "")
    st.markdown(f'<div class="preview-box">{preview_text}</div>', unsafe_allow_html=True)

    # Quick stats
    if len(text.split()) > 10:
        top_words = extract_top_keywords(text, 5)
        st.caption(f"**Top keywords:** {', '.join([f'{w} ({c})' for w, c in top_words])}")

    # -----------------------------
    # Classification Button
    # -----------------------------
    if st.button("🔍 Classify Document", use_container_width=True, type="primary"):
        start_time = time.time()
        
        with st.spinner("AI is analyzing..."):
            if debug_mode:
                predicted_label, confidence, scores, matched_keywords, is_other, other_reason, all_matches = classify_document(
                    text, slider_threshold, debug=True
                )
            else:
                predicted_label, confidence, scores, matched_keywords, is_other, other_reason = classify_document(
                    text, slider_threshold
                )
            
            processing_time = time.time() - start_time

        # Save to history
        st.session_state.history.append({
            'filename': uploaded_file.name,
            'category': predicted_label,
            'confidence': confidence,
            'is_other': is_other,
            'time': datetime.now().strftime("%H:%M:%S"),
            'words': len(text.split())
        })
        st.session_state.classification_count += 1

        # Display result
        st.subheader("🎯 Classification Result")

        box_class = "result-other" if is_other else ("result-success" if confidence > 0.7 else "result-warning")
        emoji = "🤔" if is_other else ("✅" if confidence > 0.7 else "⚠️")

        st.markdown(f'''
            <div class="result-box {box_class}">
                <h2 style="margin:0;">{emoji} {predicted_label}</h2>
                <p style="margin:0.5rem 0 0 0; color:#555;">
                    Confidence: {confidence:.1%} • ⏱️ {processing_time:.2f}s
                </p>
                {f'<p style="margin:0.3rem 0 0 0; color:#888; font-size:0.9rem;">💡 {other_reason}</p>' if is_other else ''}
            </div>
        ''', unsafe_allow_html=True)

        st.progress(min(confidence, 1.0))

        # Export results
        export_df = export_results(uploaded_file, predicted_label, confidence, scores, text)
        csv = export_df.to_csv(index=False).encode('utf-8')
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # -----------------------------
        # Detailed Analysis
        # -----------------------------
        with st.expander("📊 Detailed Analysis", expanded=True):
            # Category scores with better visualization
            st.write("**Category Scores:**")
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            max_score = max(scores.values()) if scores.values() else 1
            for category, score in sorted_scores:
                bar_width = score / max_score if max_score > 0 else 0
                icon = CATEGORIES.get(category, {}).get("icon", "📄")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{icon} **{category}**")
                    st.progress(bar_width)
                with col2:
                    st.write(f"{score} matches")

            st.write("**Keywords Found:**")
            found_any = False
            for category, keywords in matched_keywords.items():
                if keywords:
                    found_any = True
                    st.write(f"**{category}:**")
                    st.write(' '.join([f'<span class="category-chip">{k}</span>' for k in keywords[:10]]), unsafe_allow_html=True)

            if not found_any:
                st.write("No keywords matched")

            # Document summary
            st.write("**Document Summary:**")
            summary = get_document_summary(text)
            st.write(summary)

            # Debug information
            if debug_mode:
                st.subheader("🐞 Debug Information")
                
                st.write("**Detailed Keyword Matches:**")
                for category, matches in all_matches.items():
                    if matches:
                        st.write(f"**{category}:**")
                        for keyword, count in list(matches.items())[:10]:
                            st.write(f"  - '{keyword}': {count} time(s)")
                
                st.write("**Text Sample (first 500 chars):**")
                st.code(text[:500], language='text')
                
                st.write("**Word Frequency:**")
                top_words = extract_top_keywords(text, 15)
                for word, count in top_words:
                    st.write(f"  - '{word}': {count}")

        # History stats
        if st.session_state.history:
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Classifications", st.session_state.classification_count)
            with col2:
                avg_conf = sum(h['confidence'] for h in st.session_state.history) / len(st.session_state.history)
                st.metric("🎯 Avg Confidence", f"{avg_conf:.1%}")

else:
    # Empty state with better design
    st.info("👆 Upload a .txt or .pdf file to begin classification")
    st.caption("Supported categories: " + ", ".join(CATEGORIES.keys()))
    st.caption("Documents that don't match any category will be marked as 'Other'")

# -----------------------------
# History (if exists)
# -----------------------------
if st.session_state.history:
    with st.expander("📜 Classification History"):
        st.write(f"Total: {st.session_state.classification_count} documents")
        
        # Show last 5
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            st.write(f"{i+1}. {item['category']} - {item['confidence']:.1%} ({item['time']})")

st.divider()
st.caption("Built with 🤗 Transformers • Powered by DistilBERT")