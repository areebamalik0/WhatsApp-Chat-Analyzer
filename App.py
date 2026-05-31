import streamlit as st
import joblib
import re
import nltk
import emoji
import base64
from collections import Counter
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

#loading models
tox_model = joblib.load("toxicity_model.pkl")
tox_vec = joblib.load("toxicity_vectorizer.pkl")

english_words = set(nltk.corpus.words.words())
sia = SentimentIntensityAnalyzer()


def parse_chat(file):
    users, messages = [], []
    for line in file.readlines():
        try:
            line = line.decode("utf-8")
        except:
            pass
        if " - " in line and ": " in line:
            user, msg = line.split(": ",1) 
            user = user.split(" - ",1)[1] 
            users.append(user)
            messages.append(msg.strip()) 
    return users, messages

def clean_text_for_toxicity(text):
    text = str(text).lower()
    text = re.sub(r"<media omitted>|media omitted", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def analyze_chat(messages, users):
    total_words = 0
    english_word_count = 0
    for msg in messages:
        msg_clean = re.sub(r'[^a-z\s]', '', msg.lower())
        words = msg_clean.split()
        for w in words:
            if len(w) > 1:
                total_words += 1
                if w in english_words:
                    english_word_count += 1
    if total_words == 0 or (english_word_count / total_words) < 0.5:
        st.warning("⚠️ This chat contains mostly non-English text. The app currently supports English only.")
        st.stop()

    user_counts = Counter(users)
    most_active_user = user_counts.most_common(1)[0]

    all_emojis = []
    for msg in messages:
        all_emojis.extend([c for c in msg if c in emoji.EMOJI_DATA])
    emoji_count = Counter(all_emojis)
    top_emoji, top_emoji_count = emoji_count.most_common(1)[0] if emoji_count else ("None",0)

    words = []
    for msg in messages:
      msg_clean = re.sub(r"<media omitted>|media omitted", '', msg.lower())
      msg_clean = re.sub(r"[^a-zA-Z\s]", "", msg_clean)  
      words.extend(msg_clean.split())
    most_common_word = Counter(words).most_common(1)[0]

    messages_cleaned = [clean_text_for_toxicity(msg) for msg in messages]
    X_tox = tox_vec.transform(messages_cleaned)

    tox_pred_proba = tox_model.predict_proba(X_tox)
    tox_pred = [1 if p[1]>=0.5 else 0 for p in tox_pred_proba]
    toxic_pct = (sum(tox_pred)/len(tox_pred))*100
    non_toxic_pct = 100 - toxic_pct

    pos = neg = neu = 0
    for msg in messages:
        scores = sia.polarity_scores(msg)
        compound = scores['compound']

        if compound >= 0.05:
            pos += 1
        elif compound <= -0.05:
            neg += 1
        else:
            neu += 1

    total = len(messages)
    pos_pct = (pos/total)*100
    neg_pct = (neg/total)*100
    neu_pct = (neu/total)*100

    return {
        "most_active_user": most_active_user,
        "top_emoji": (top_emoji,top_emoji_count),
        "most_common_word": most_common_word,
        "toxic_pct": toxic_pct,
        "non_toxic_pct": non_toxic_pct,
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "neu_pct": neu_pct
    }

# Page setup
st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide", page_icon="💬")

# Custom styles (dark WhatsApp theme)
st.markdown("""
    <style>
    .main {
        background-color: #121b22;
        padding: 2rem;
        color: white;
    }
    .metric-card {
        background-color: #1e2a33;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        margin: 1rem 0;
        text-align: center;
        color: white;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #25D366;
    }
    .metric-label {
        font-size: 1rem;
        color: #cccccc;
        margin-top: 0.5rem;
    }
    h1 {
        color: #25D366;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header with logo (LEFT ALIGNED)
with open("whatsapp.png", "rb") as f:
    logo_data = f.read()
logo_b64 = base64.b64encode(logo_data).decode()

st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:12px;">
        <img src="data:image/png;base64,{logo_b64}" width="90"/>
        <span style="font-size:52px; font-weight:bold; color:#FFFFFF;">
            WhatsApp Chat Analyzer
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
st.markdown("### 📤 Upload Your Chat File")
uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"], help="Export your WhatsApp chat as .txt file")

if uploaded_file:
    with st.spinner("Analyzing your chat..."):
        users, messages = parse_chat(uploaded_file)
        results = analyze_chat(messages, users)

    if results:
        st.success(f"✅ Analyzed {len(messages)} messages successfully!")
        
        st.markdown("### Chat Statistics")
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{results['most_active_user'][1]}</div>
                    <div class="metric-label">Messages by {results['most_active_user'][0]}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with stats_col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size:3rem;">{results['top_emoji'][0]}</div>
                    <div class="metric-label">Used {results['top_emoji'][1]} times</div>
                </div>
            """, unsafe_allow_html=True)
        
        with stats_col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size:2rem;">{results['most_common_word'][0]}</div>
                    <div class="metric-label">Most frequent word ({results['most_common_word'][1]}x)</div>
                </div>
            """, unsafe_allow_html=True)
        
        with stats_col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#F74B42;">{results['toxic_pct']:.1f}%</div>
                    <div class="metric-label">Toxic Content</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📈 Detailed Analysis")
        analysis_col1, analysis_col2 = st.columns([1, 1.5])
        
        with analysis_col1:
            st.markdown("#### 🎭 Sentiment Breakdown")
            st.metric("Positive", f"{results['pos_pct']:.1f}%", delta=f"{results['pos_pct']:.1f}%")
            st.metric("Negative", f"{results['neg_pct']:.1f}%", delta=f"-{results['neg_pct']:.1f}%")
            st.metric("Neutral", f"{results['neu_pct']:.1f}%")
            
            st.markdown("#### ⚠️ Toxicity Analysis")
            st.metric("Toxic", f"{results['toxic_pct']:.1f}%", delta=f"{results['toxic_pct']:.1f}%")
            st.metric("Non-Toxic", f"{results['non_toxic_pct']:.1f}%")
        
        with analysis_col2:
            fig, ax = plt.subplots(figsize=(10, 6))

            # DARK GRAPH
            fig.patch.set_facecolor("#121b22")
            ax.set_facecolor("#0e1117")

            ax.tick_params(colors="white", labelsize=12)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

            for spine in ax.spines.values():
                spine.set_color("white")

            categories = ["Toxic", "Non-Toxic", "Positive", "Negative", "Neutral"]
            percentages = [results['toxic_pct'], results['non_toxic_pct'], 
                          results['pos_pct'], results['neg_pct'], results['neu_pct']]
            colors = ["#F74B42", "#34D65D", "#2F8DDA", "#FF309B", "#7F7FD4"]

            bars = ax.barh(categories, percentages, color=colors)

            ax.set_xlim(0, 100)
            ax.set_xlabel("Percentage (%)", fontsize=12, fontweight='bold')
            ax.set_title("Complete Chat Analysis", fontsize=16, fontweight='bold', pad=20)

            ax.grid(axis='x', alpha=0.2)

            for i, (bar, pct) in enumerate(zip(bars, percentages)):
                ax.text(pct + 1, i, f'{pct:.1f}%', va='center', color="white", fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)
