import streamlit as st
import time
import re
from collections import deque
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob
import pandas as pd
import math

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LiveSense – Real-Time Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0a0a0f;
    color: #e2e8f0;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f1a 100%);
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2rem 0 1.5rem;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.hero p {
    color: #64748b;
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 400;
}
.pulse-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #34d399;
    margin-right: 6px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* ── Glass cards ── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}

/* ── Sentiment banner ── */
.sentiment-banner {
    border-radius: 16px;
    padding: 1.2rem 1.8rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 1rem;
    transition: all 0.4s ease;
}
.banner-positive {
    background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(16,185,129,0.08));
    border: 1px solid rgba(52,211,153,0.3);
    color: #34d399;
}
.banner-negative {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
    border: 1px solid rgba(239,68,68,0.3);
    color: #f87171;
}
.banner-neutral {
    background: linear-gradient(135deg, rgba(148,163,184,0.12), rgba(100,116,139,0.06));
    border: 1px solid rgba(148,163,184,0.2);
    color: #94a3b8;
}
.banner-mixed {
    background: linear-gradient(135deg, rgba(251,191,36,0.15), rgba(245,158,11,0.08));
    border: 1px solid rgba(251,191,36,0.3);
    color: #fbbf24;
}

/* ── Stat boxes ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.stat-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-value {
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1;
}
.stat-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* ── Word highlights ── */
.word-display {
    line-height: 2.2;
    font-size: 1.1rem;
    padding: 1rem;
    min-height: 60px;
}
.word-pos {
    background: rgba(52,211,153,0.2);
    color: #34d399;
    border-radius: 5px;
    padding: 2px 6px;
    margin: 2px;
    display: inline-block;
    border-bottom: 2px solid #34d399;
    font-weight: 600;
}
.word-neg {
    background: rgba(248,113,113,0.2);
    color: #f87171;
    border-radius: 5px;
    padding: 2px 6px;
    margin: 2px;
    display: inline-block;
    border-bottom: 2px solid #f87171;
    font-weight: 600;
}
.word-neutral {
    color: #94a3b8;
    padding: 2px 4px;
    margin: 1px;
    display: inline-block;
}

/* ── Textarea override ── */
.stTextArea textarea {
    background: rgba(15,20,35,0.8) !important;
    border: 1px solid rgba(129,140,248,0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1.05rem !important;
    line-height: 1.7 !important;
    padding: 1rem !important;
    transition: border-color 0.3s ease !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: rgba(129,140,248,0.7) !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.1) !important;
}

/* ── Section headers ── */
.section-header {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Meter bar ── */
.meter-container {
    background: rgba(255,255,255,0.05);
    border-radius: 100px;
    height: 12px;
    overflow: hidden;
    margin: 0.5rem 0;
}
.meter-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1), background 0.5s ease;
}

/* ── Sentence breakdown ── */
.sentence-item {
    padding: 0.6rem 0.8rem;
    border-radius: 8px;
    margin-bottom: 0.4rem;
    font-size: 0.88rem;
    border-left: 3px solid transparent;
}
.sent-pos { background: rgba(52,211,153,0.08); border-left-color: #34d399; }
.sent-neg { background: rgba(248,113,113,0.08); border-left-color: #f87171; }
.sent-neu { background: rgba(148,163,184,0.06); border-left-color: #475569; }

/* ── Emoji indicator ── */
.emoji-big {
    font-size: 3rem;
    line-height: 1;
    filter: drop-shadow(0 0 12px rgba(255,255,255,0.2));
}

/* ── Auto-refresh button ── */
.stButton button {
    background: linear-gradient(135deg, #818cf8, #a78bfa) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.3s ease !important;
}
.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(129,140,248,0.35) !important;
}

/* ── Slider override ── */
.stSlider { padding: 0 !important; }
.stSlider [data-baseweb="slider"] { padding: 0.5rem 0 !important; }

/* ── Tabs override ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.4rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #818cf8, #a78bfa) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ── Positive / Negative word lexicons (lightweight, no NLTK required) ──────────
POSITIVE_WORDS = {
    "good","great","excellent","amazing","wonderful","fantastic","awesome","love",
    "happy","joy","beautiful","perfect","brilliant","outstanding","superb","nice",
    "enjoy","liked","best","incredible","magnificent","positive","delightful","glad",
    "pleased","grateful","thankful","cheerful","excited","thrilled","impressive",
    "remarkable","stunning","marvelous","splendid","terrific","fabulous","success",
    "win","winning","won","fun","laugh","smile","hope","peaceful","kind","caring",
    "sweet","generous","helpful","friendly","warm","bright","clear","fresh",
    "innovative","creative","inspiring","motivated","confident","strong","proud",
    "achieve","accomplished","effective","efficient","powerful","intelligent",
    "smart","wise","honest","genuine","authentic","trustworthy","reliable",
}
NEGATIVE_WORDS = {
    "bad","terrible","awful","horrible","disgusting","hate","sad","angry","upset",
    "disappointed","frustrated","annoyed","furious","depressed","miserable","worst",
    "failure","fail","failed","wrong","broken","useless","boring","dull","ugly",
    "evil","corrupt","dishonest","lazy","stupid","idiot","dumb","pathetic","weak",
    "pain","hurt","suffer","cry","fear","scared","worried","anxious","stressed",
    "problem","issue","trouble","difficult","hard","impossible","never","nothing",
    "nobody","nowhere","terrible","disastrous","catastrophic","devastating","ruin",
    "destroy","damage","loss","lose","lost","dying","dead","alone","lonely","empty",
    "hopeless","helpless","worthless","meaningless","pointless","disgusted","enraged",
}

def clean_word(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower())

def analyze_text(text: str) -> dict:
    """Full sentiment analysis of the given text."""
    if not text or not text.strip():
        return {
            "polarity": 0.0, "subjectivity": 0.0,
            "label": "neutral", "emoji": "💬",
            "words": [], "sentences": [],
            "word_count": 0, "char_count": 0,
            "pos_count": 0, "neg_count": 0,
            "mood_history": [],
        }

    blob = TextBlob(text)
    polarity    = blob.sentiment.polarity       # -1 → +1
    subjectivity = blob.sentiment.subjectivity  # 0 → 1

    # Determine label & emoji
    if polarity > 0.3:
        label, emoji = "positive", "😊"
    elif polarity > 0.05:
        label, emoji = "positive", "🙂"
    elif polarity < -0.3:
        label, emoji = "negative", "😠"
    elif polarity < -0.05:
        label, emoji = "negative", "😕"
    else:
        label, emoji = "neutral", "😐"

    # Word-level tagging
    raw_words = text.split()
    tagged_words = []
    pos_count = neg_count = 0
    for w in raw_words:
        cw = clean_word(w)
        if cw in POSITIVE_WORDS:
            tagged_words.append(("pos", w))
            pos_count += 1
        elif cw in NEGATIVE_WORDS:
            tagged_words.append(("neg", w))
            neg_count += 1
        else:
            tagged_words.append(("neu", w))

    # Sentence-level analysis
    sentences = []
    for sent in blob.sentences:
        s = sent.raw
        sp = sent.sentiment.polarity
        if sp > 0.05:
            slabel = "pos"
        elif sp < -0.05:
            slabel = "neg"
        else:
            slabel = "neu"
        sentences.append({"text": s, "polarity": sp, "label": slabel})

    # Mixed detection
    if pos_count > 0 and neg_count > 0:
        label = "mixed"
        emoji = "🤔"

    return {
        "polarity": round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
        "label": label,
        "emoji": emoji,
        "words": tagged_words,
        "sentences": sentences,
        "word_count": len([w for w in raw_words if w.strip()]),
        "char_count": len(text),
        "pos_count": pos_count,
        "neg_count": neg_count,
    }


def polarity_to_pct(polarity: float) -> int:
    """Map -1..+1 to 0..100"""
    return int((polarity + 1) / 2 * 100)

def make_gauge(polarity: float, label: str):
    """Create a Plotly gauge chart for polarity."""
    if label == "positive":
        color = "#34d399"
    elif label == "negative":
        color = "#f87171"
    elif label == "mixed":
        color = "#fbbf24"
    else:
        color = "#94a3b8"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(polarity, 3),
        number={
            "suffix": "",
            "font": {"size": 28, "color": color, "family": "Inter"},
            "valueformat": ".3f",
        },
        gauge={
            "axis": {
                "range": [-1, 1],
                "tickwidth": 1,
                "tickcolor": "#1e293b",
                "tickvals": [-1, -0.5, 0, 0.5, 1],
                "ticktext": ["−1", "−0.5", "0", "+0.5", "+1"],
                "tickfont": {"color": "#475569", "size": 11},
            },
            "bar": {"color": color, "thickness": 0.6},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [-1, -0.3], "color": "rgba(239,68,68,0.12)"},
                {"range": [-0.3, 0.3], "color": "rgba(148,163,184,0.07)"},
                {"range": [0.3, 1],   "color": "rgba(52,211,153,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": polarity,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=10),
        height=200,
        font={"family": "Inter"},
    )
    return fig


def make_trend_chart(history: list):
    """Line chart of polarity over typing history snapshots."""
    if len(history) < 2:
        return None
    df = pd.DataFrame(history, columns=["chars", "polarity"])

    fig = go.Figure()
    # Fill area
    fig.add_trace(go.Scatter(
        x=df["chars"], y=df["polarity"],
        fill="tozeroy",
        fillcolor="rgba(129,140,248,0.07)",
        line=dict(color="rgba(129,140,248,0)", width=0),
        showlegend=False, hoverinfo="skip",
    ))
    # Main line
    colors = []
    for p in df["polarity"]:
        if p > 0.05:
            colors.append("#34d399")
        elif p < -0.05:
            colors.append("#f87171")
        else:
            colors.append("#94a3b8")

    fig.add_trace(go.Scatter(
        x=df["chars"], y=df["polarity"],
        mode="lines+markers",
        line=dict(color="#818cf8", width=2.5, shape="spline", smoothing=0.8),
        marker=dict(
            size=6, color=colors,
            line=dict(color="#0a0a0f", width=1.5),
        ),
        hovertemplate="<b>Chars:</b> %{x}<br><b>Polarity:</b> %{y:.3f}<extra></extra>",
        name="Polarity",
    ))
    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(148,163,184,0.2)", line_width=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=15, b=10),
        height=180,
        showlegend=False,
        xaxis=dict(
            title="Characters typed",
            color="#475569", gridcolor="rgba(255,255,255,0.04)",
            title_font_size=11, tickfont_size=10,
        ),
        yaxis=dict(
            range=[-1.1, 1.1],
            title="Sentiment",
            color="#475569", gridcolor="rgba(255,255,255,0.04)",
            title_font_size=11, tickfont_size=10,
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["−1", "−0.5", "0", "+0.5", "+1"],
        ),
        font={"family": "Inter"},
    )
    return fig


def make_word_bar(pos_count: int, neg_count: int, neu_count: int):
    categories = ["Positive", "Negative", "Neutral"]
    values     = [pos_count, neg_count, neu_count]
    clrs       = ["#34d399", "#f87171", "#94a3b8"]

    fig = go.Figure(go.Bar(
        x=categories, y=values,
        marker_color=clrs,
        marker_line_width=0,
        text=values,
        textposition="outside",
        textfont=dict(size=13, color="#e2e8f0"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=5, t=15, b=5),
        height=180,
        xaxis=dict(color="#475569", tickfont_size=11),
        yaxis=dict(
            color="#475569", gridcolor="rgba(255,255,255,0.04)",
            tickfont_size=10,
        ),
        font={"family": "Inter"},
        showlegend=False,
    )
    return fig


# ── Session state ───────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of (char_count, polarity)
if "prev_text" not in st.session_state:
    st.session_state.prev_text = ""
if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = 0

SAMPLE_TEXTS = [
    "I absolutely love this! The experience has been wonderful and incredibly positive.",
    "This is terrible. I'm so frustrated and disappointed with everything going wrong.",
    "The weather today is okay. I went to the store and bought some groceries.",
    "I'm happy about some things but also worried about others. It's a mixed bag.",
    "The technology is brilliant and innovative, though some aspects are still problematic.",
]

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 LiveSense</h1>
    <p><span class="pulse-dot"></span>Real-time sentiment analysis — analysed as you type, character by character</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ──────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    # ── Input area ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">✏️ Type your text below</div>', unsafe_allow_html=True)
    
    user_text = st.text_area(
        label="typing_input",
        label_visibility="collapsed",
        placeholder="Start typing anything... your mood will be analysed in real time ✨",
        height=200,
        key="live_input",
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn1:
        if st.button("🎲 Load Sample", use_container_width=True):
            idx = st.session_state.sample_idx % len(SAMPLE_TEXTS)
            st.session_state.sample_idx += 1
            # We use st.session_state to pre-fill — workaround via rerun
            st.session_state["_sample_text"] = SAMPLE_TEXTS[idx]
            st.rerun()
    with col_btn2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.history = []
            st.session_state.prev_text = ""
            st.session_state["_sample_text"] = ""
            st.rerun()
    with col_btn3:
        auto_refresh = st.toggle("⚡ Live mode", value=True)

    # Handle sample injection
    if "_sample_text" in st.session_state and st.session_state["_sample_text"]:
        user_text = st.session_state["_sample_text"]
        del st.session_state["_sample_text"]

    # ── Analysis ────────────────────────────────────────────────────────────────
    result = analyze_text(user_text)

    # Track history every time char count changes by ≥3
    char_now = result["char_count"]
    if (char_now != st.session_state.prev_text and char_now > 0 and
            (not st.session_state.history or
             abs(char_now - st.session_state.history[-1][0]) >= 3)):
        st.session_state.history.append((char_now, result["polarity"]))
        st.session_state.prev_text = char_now
    # Cap history at 120 points
    if len(st.session_state.history) > 120:
        st.session_state.history = st.session_state.history[-120:]

    # ── Sentiment banner ─────────────────────────────────────────────────────────
    label   = result["label"]
    emoji   = result["emoji"]
    polarity = result["polarity"]
    bclass  = f"banner-{label}"

    label_text_map = {
        "positive": "Positive Sentiment",
        "negative": "Negative Sentiment",
        "neutral":  "Neutral Sentiment",
        "mixed":    "Mixed Sentiment",
    }
    if user_text and user_text.strip():
        st.markdown(f"""
        <div class="sentiment-banner {bclass}">
            <span class="emoji-big">{emoji}</span>
            <div>
                <div style="font-size:1.3rem;font-weight:800;">{label_text_map[label]}</div>
                <div style="font-size:0.85rem;opacity:0.7;font-weight:400;margin-top:2px;">
                    Polarity: {polarity:+.3f} &nbsp;·&nbsp; Subjectivity: {result['subjectivity']:.0%}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sentiment-banner banner-neutral">
            <span class="emoji-big">💬</span>
            <div>
                <div style="font-size:1.1rem;font-weight:600;color:#475569;">
                    Waiting for input — start typing to see live analysis
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Stats ────────────────────────────────────────────────────────────────────
    wc   = result["word_count"]
    cc   = result["char_count"]
    pc   = result["pos_count"]
    nc   = result["neg_count"]
    sents = result["sentences"]

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-box">
            <div class="stat-value" style="color:#818cf8;">{wc}</div>
            <div class="stat-label">Words</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" style="color:#38bdf8;">{cc}</div>
            <div class="stat-label">Characters</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" style="color:#34d399;">{pc}</div>
            <div class="stat-label">Positive Words</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" style="color:#f87171;">{nc}</div>
            <div class="stat-label">Negative Words</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Word-level highlight ─────────────────────────────────────────────────────
    if result["words"]:
        st.markdown('<div class="section-header">🔬 Word-level Analysis</div>', unsafe_allow_html=True)
        word_html = '<div class="glass-card word-display">'
        for tag, word in result["words"]:
            esc = word.replace("<", "&lt;").replace(">", "&gt;")
            if tag == "pos":
                word_html += f'<span class="word-pos">✓ {esc}</span> '
            elif tag == "neg":
                word_html += f'<span class="word-neg">✗ {esc}</span> '
            else:
                word_html += f'<span class="word-neutral">{esc}</span> '
        word_html += "</div>"
        st.markdown(word_html, unsafe_allow_html=True)

    # ── Sentence breakdown ───────────────────────────────────────────────────────
    if len(sents) > 1:
        st.markdown('<div class="section-header">📝 Sentence Breakdown</div>', unsafe_allow_html=True)
        for s in sents[:8]:
            icon = {"pos": "😊", "neg": "😔", "neu": "😐"}[s["label"]]
            cls  = f"sent-{s['label']}"
            esc  = s["text"][:120].replace("<","&lt;").replace(">","&gt;")
            st.markdown(f"""
            <div class="sentence-item {cls}">
                {icon} <b>{s['polarity']:+.3f}</b> &nbsp; {esc}{"…" if len(s["text"]) > 120 else ""}
            </div>
            """, unsafe_allow_html=True)


with right_col:
    st.markdown('<div class="section-header">📊 Sentiment Gauge</div>', unsafe_allow_html=True)
    gauge_fig = make_gauge(polarity, label)
    st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})  # noqa

    # ── Polarity meter bar ───────────────────────────────────────────────────────
    pct = polarity_to_pct(polarity)
    if label == "positive":
        bar_color = "linear-gradient(90deg, #059669, #34d399)"
    elif label == "negative":
        bar_color = "linear-gradient(90deg, #dc2626, #f87171)"
    elif label == "mixed":
        bar_color = "linear-gradient(90deg, #d97706, #fbbf24)"
    else:
        bar_color = "linear-gradient(90deg, #334155, #64748b)"

    st.markdown(f"""
    <div style="margin-bottom:1.2rem;">
        <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#475569;margin-bottom:4px;">
            <span>Very Negative</span><span>Neutral</span><span>Very Positive</span>
        </div>
        <div class="meter-container">
            <div class="meter-fill" style="width:{pct}%;background:{bar_color};"></div>
        </div>
        <div style="text-align:center;font-size:0.75rem;color:#64748b;margin-top:4px;">{pct}% positive</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Subjectivity meter ───────────────────────────────────────────────────────
    subj = result["subjectivity"]
    subj_pct = int(subj * 100)
    st.markdown(f"""
    <div style="margin-bottom:1.4rem;">
        <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#475569;margin-bottom:4px;">
            <span>Objective</span><span style="text-align:center;">Subjectivity</span><span>Subjective</span>
        </div>
        <div class="meter-container">
            <div class="meter-fill" style="width:{subj_pct}%;background:linear-gradient(90deg,#4f46e5,#818cf8);"></div>
        </div>
        <div style="text-align:center;font-size:0.75rem;color:#64748b;margin-top:4px;">{subj_pct}% subjective</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs: trend / word dist ──────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📈 Mood Trend", "📊 Word Distribution"])

    with tab1:
        trend_fig = make_trend_chart(st.session_state.history)
        if trend_fig:
            st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})  # noqa
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center;color:#475569;padding:2rem;font-size:0.9rem;">
                📈 Mood trend will appear as you type more...
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        neu_count = result["word_count"] - result["pos_count"] - result["neg_count"]
        bar_fig = make_word_bar(result["pos_count"], result["neg_count"], max(0, neu_count))
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})  # noqa

    # ── Tips panel ───────────────────────────────────────────────────────────────
    if user_text and user_text.strip():
        tips = []
        if polarity > 0.5:
            tips.append(("💚", "Very positive tone detected! Great energy."))
        elif polarity < -0.5:
            tips.append(("❤️", "Strong negative tone. Consider rephrasing for a softer impact."))
        if subj > 0.7:
            tips.append(("🧠", "Highly subjective — this reads as a personal opinion."))
        elif subj < 0.2:
            tips.append(("📋", "Very objective and factual writing style."))
        if result["pos_count"] > 0 and result["neg_count"] > 0:
            tips.append(("⚖️", "Mixed sentiments — both positive and negative words present."))
        if wc > 0 and wc < 5:
            tips.append(("📝", "Type more for a more accurate analysis."))

        if tips:
            st.markdown('<div class="section-header" style="margin-top:0.5rem;">💡 Insights</div>', unsafe_allow_html=True)
            for icon, tip in tips:
                st.markdown(f"""
                <div class="glass-card" style="padding:0.6rem 0.9rem;font-size:0.85rem;color:#cbd5e1;margin-bottom:0.4rem;">
                    {icon} {tip}
                </div>
                """, unsafe_allow_html=True)


# ── Auto-refresh for live mode ────────────────────────────────────────────────
if auto_refresh:
    time.sleep(0.8)
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#1e293b;font-size:0.75rem;margin-top:2rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.04);">
    LiveSense &nbsp;·&nbsp; Real-time sentiment analysis powered by TextBlob &amp; Plotly
</div>
""", unsafe_allow_html=True)
