import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import re

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
    .main { padding: 2rem; }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover { background-color: #45a049; }
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

# Document categories with keywords
CATEGORIES = {
    "📧 Email": {
        "keywords": ["dear", "hello", "hi", "regards", "sincerely", "thanks", "thank you", "email", "sent", "received", 
                    "subject", "reply", "forward", "cc", "bcc", "attachment", "greetings", "best regards", "kind regards"],
        "icon": "📧",
        "color": "#2196F3",
        "threshold": 3  # Minimum matches needed
    },
    "📄 Invoice": {
        "keywords": ["invoice", "payment", "amount", "due", "total", "balance", "receipt", "billing", "purchase", "order",
                    "subtotal", "tax", "discount", "price", "cost", "paid", "outstanding", "statement", "transaction"],
        "icon": "📄",
        "color": "#FF9800",
        "threshold": 3
    },
    "📋 Report": {
        "keywords": ["report", "analysis", "summary", "findings", "results", "data", "conclusion", "recommendation",
                    "performance", "metrics", "kpi", "dashboard", "trend", "forecast", "evaluation", "assessment"],
        "icon": "📋",
        "color": "#9C27B0",
        "threshold": 3
    },
    "📝 Resume": {
        "keywords": ["experience", "skills", "education", "job", "work", "career", "position", "accomplishments",
                    "professional", "certification", "degree", "university", "employment", "responsibilities",
                    "achievements", "references", "objective", "summary", "qualifications"],
        "icon": "📝",
        "color": "#4CAF50",
        "threshold": 3
    },
    "📑 Contract": {
        "keywords": ["agreement", "terms", "conditions", "party", "clause", "effective", "signature", "legal",
                    "obligation", "liability", "indemnify", "warranty", "termination", "confidential", "governing",
                    "jurisdiction", "arbitration", "force majeure", "non-disclosure"],
        "icon": "📑",
        "color": "#F44336",
        "threshold": 3
    },
    "💻 Code": {
        "keywords": ["import", "def", "class", "function", "return", "print", "if", "else", "for", "while",
                    "try", "except", "with", "as", "from", "lambda", "async", "await", "yield", "global",
                    "nonlocal", "assert", "pass", "break", "continue", "raise", "self", "super"],
        "icon": "💻",
        "color": "#607D8B",
        "threshold": 3
    }
}

def classify_document(text):
    """Classify document using keyword matching with threshold and fallback"""
    text_lower = text.lower()
    text_words = set(text_lower.split())  # For additional analysis
    
    # Detect if it's code
    code_patterns = re.findall(r'\b(import|def|class|function|return|print|if|else|for|while|try|except)\b', text_lower)
    has_code = len(code_patterns) > 5
    
    scores = {}
    matched_keywords = {}
    unique_matches = {}  # Track unique keywords matched
    
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
        
        # Store unique matches count
        unique_matches[category] = unique_count
        
        # Boost score if multiple matches from same category
        if len(matches) > 2:
            score *= 1.5
        
        scores[category] = score
        matched_keywords[category] = matches
    
    # Code detection override
    if has_code and scores["💻 Code"] > 0:
        scores["💻 Code"] *= 2
    
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
    if word_count > 20 and max_score / word_count < 0.05:  # Less than 5% of words match
        is_other = True
        other_reason = f"Very few keyword matches ({max_score} matches in {word_count} words)"
    
    # Condition 3: Multiple categories have similar scores (ambiguous)
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
    
    # If no keywords found at all, use sentiment analysis
    if max_score == 0:
        inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits).item()
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities[0][predicted_class].item()
            confidence = min(confidence, 0.4)  # Reduce confidence for sentiment fallback
        
        # Map sentiment to categories with low confidence
        if predicted_class == 1:  # Positive
            predicted_label = "📄 Invoice"
        else:  # Negative
            predicted_label = "📧 Email"
        
        # Mark as other if confidence is too low
        if confidence < 0.3:
            is_other = True
            other_reason = "Low confidence sentiment analysis fallback"
    
    # If marked as other, change label
    if is_other:
        predicted_label = "❓ Other / Unknown"
        confidence = max(confidence, 0.1)  # Ensure some visibility
    
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
    st.header("📌 Supported Types")
    for category, info in CATEGORIES.items():
        st.write(f"{info['icon']} {category}")

# Create upload section
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
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
    
    # File info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📁 File Name", uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)
    with col2:
        st.metric("📏 Size", f"{len(text):,} chars")
    with col3:
        st.metric("📝 Words", len(text.split()))
    
    # Document preview
    st.subheader("📖 Document Preview")
    preview_text = text[:500] + ("..." if len(text) > 500 else "")
    st.markdown(f'<div class="preview-box">{preview_text}</div>', unsafe_allow_html=True)
    
    # Classification button
    if st.button("🔍 Classify Document", use_container_width=True):
        with st.spinner("AI is analyzing..."):
            predicted_label, confidence, scores, matched_keywords, is_other, other_reason = classify_document(text)
        
        # Display result with appropriate styling
        st.subheader("🎯 Classification Result")
        
        # Choose result box style
        if is_other:
            box_class = "result-other"
            emoji = "🤔"
        elif confidence > 0.7:
            box_class = "result-success"
            emoji = "✅"
        else:
            box_class = "result-warning"
            emoji = "⚠️"
        
        st.markdown(f'''
            <div class="result-box {box_class}">
                <h2 style="margin:0;">{emoji} {predicted_label}</h2>
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
            # Show scores for all categories
            st.write("**Category Scores:**")
            
            # Sort categories by score
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
            
            # Show matched keywords
            st.write("**Keywords Found:**")
            found_any = False
            for category, keywords in matched_keywords.items():
                if keywords:
                    found_any = True
                    st.write(f"**{category}:** {', '.join(keywords[:10])}")
            
            if not found_any:
                st.write("No keywords matched")
            
            # Show detected code patterns
            code_patterns = re.findall(r'\b(import|def|class|function|return|print|if|else|for|while|try|except)\b', text.lower())
            if code_patterns:
                st.info(f"💻 **Code detected!** Found these patterns: {', '.join(set(code_patterns[:10]))}")
            
            # Show document statistics
            st.write("**Document Statistics:**")
            words = text.split()
            st.write(f"- Total words: {len(words)}")
            st.write(f"- Unique words: {len(set(words))}")
            st.write(f"- Average word length: {sum(len(w) for w in words) / len(words) if words else 0:.1f} characters")
else:
    # Empty state
    st.info("👆 Upload a .txt file to begin classification")
    st.caption("Supported categories: Email, Invoice, Report, Resume, Contract, Code")
    st.caption("Documents that don't match any category will be marked as 'Other'")

# Footer
st.divider()
st.caption("Built with 🤗 Transformers • Powered by DistilBERT")