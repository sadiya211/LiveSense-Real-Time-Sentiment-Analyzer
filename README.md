# 🧠 LiveSense – Real-Time Sentiment Analyzer

> Analyse the emotion and sentiment of your text **as you type**, character by character — in real time.

![LiveSense Banner](https://img.shields.io/badge/LiveSense-Real--Time%20Sentiment-818cf8?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![TextBlob](https://img.shields.io/badge/TextBlob-NLP-34d399?style=for-the-badge)

---

## 🚀 Features

### 🔴 Live Analysis
- ⚡ **Real-time updates** every 0.8 seconds as you type (toggleable Live Mode)
- 😊 **Sentiment banner** — Positive / Negative / Neutral / Mixed with dynamic colour theming
- 📊 **Polarity score** from −1.0 (very negative) to +1.0 (very positive)
- 🎚️ **Subjectivity meter** — Objective ↔ Subjective visual bar

### 🎭 Emotion & Language
- 😊😠😢😨😌 **Emotion detection** — Happy, Angry, Sad, Fearful, Calm
- 🌍 **Language detection** — auto-detects 30+ languages with flag emoji (🇬🇧 🇮🇳 🇫🇷 🇩🇪 ...)

### 🔬 Deep Text Analysis
- **Word-level highlighting** — green for positive words, red for negative words
- **Sentence-by-sentence breakdown** with individual polarity scores
- **Stats panel** — word count, character count, positive/negative word counts

### 📈 Charts & Visualisations
- **Plotly gauge chart** showing live polarity needle
- **Mood trend chart** — tracks how sentiment shifts as you type more
- **Word distribution bar chart** — Positive vs Negative vs Neutral words

### 🛠️ Utility Tools
- 🔊 **Text-to-Speech** — reads your text aloud in the detected language (browser Web Speech API)
- 📄 **Export TXT** — download a formatted plain-text analysis report
- 📋 **Copy Results** — one-click copy of the summary to clipboard
- 🎲 **Load Sample** — cycle through 5 example texts to explore the app instantly
- 💡 **Smart Insights** — contextual tips about your writing tone

### 🎨 UI / UX
- Premium **dark-mode** design with glassmorphism cards
- Smooth gradient animations and micro-interactions
- Inter font, custom Plotly themes, fully responsive layout

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / App** | Streamlit |
| **NLP / Sentiment** | TextBlob, NLTK |
| **Language Detection** | langdetect |
| **Charts** | Plotly |
| **Data** | Pandas |
| **TTS / Clipboard** | Web Speech API (browser) |
| **Language** | Python 3.11 |

---

## ⚙️ Installation & Run

```bash
# 1. Clone the repository
git clone https://github.com/sadiya211/LiveSense-Real-Time-Sentiment-Analyzer.git
cd LiveSense-Real-Time-Sentiment-Analyzer

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download TextBlob corpora
python -m textblob.download_corpora

# 5. Run the app
streamlit run App/realtime_app.py
```

Open your browser at **http://localhost:8501**

> **Note:** On first run, NLTK data (`punkt_tab`) is automatically downloaded into the
> local `nltk_data/` folder inside the project — no manual steps needed.

---

## 📁 Project Structure

```
LiveSense-Real-Time-Sentiment-Analyzer/
│
├── App/
│   └── realtime_app.py       # Main Streamlit application
│
├── Data/                     # Sample data / datasets (if any)
│
├── nltk_data/                # Auto-created: local NLTK corpora (trusted path fix)
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 📦 Requirements

```
streamlit
textblob
plotly
pandas
nltk
langdetect
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔧 How It Works

1. **Type** anything in the text box
2. Every **0.8 seconds**, the app reads your current text
3. **TextBlob** calculates `polarity` (−1 to +1) and `subjectivity` (0 to 1)
4. **langdetect** identifies the written language (activates after 8+ words)
5. A custom **emotion lexicon** determines the dominant emotion
6. All **charts, highlights, and stats** update instantly
7. Use **🔊 Read Aloud** to hear it, **📄 Export TXT** to save it, or **📋 Copy** to share it

---

## 💡 Example Analysis

```
Input  : "I absolutely love this project, it is amazing and really inspiring!"

Output :
  Sentiment   : Positive (+0.625)
  Subjectivity: 80%
  Emotion     : Happy
  Language    : English
  Positive    : 4 words
  Negative    : 0 words
```

---

## 👤 Author

Made with ❤️ by **Anmol**

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
