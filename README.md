# LiveSense – Real-Time Typing Sentiment Analyzer 🧠

A real-time sentiment analysis web app built with **Streamlit** that analyses your text emotion **as you type**, character by character.

## 🚀 Features
- ⚡ Live sentiment updates every 0.8 seconds as you type
- 😊 Sentiment banner with emoji (Positive / Negative / Neutral / Mixed)
- 🔬 Word-level highlighting — green for positive, red for negative
- 📊 Plotly gauge chart showing polarity score (−1 to +1)
- 📈 Mood trend chart tracking sentiment as you type more
- 📝 Sentence-by-sentence breakdown
- 💡 Smart insights about your writing tone
- 🎨 Premium dark-mode UI with glassmorphism design

## 🛠️ Tech Stack
- **Python**
- **Streamlit**
- **TextBlob** (NLP)
- **Plotly** (Charts)
- **Pandas**

## ⚙️ Installation & Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/livesense-sentiment-analyzer.git
cd livesense-sentiment-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download TextBlob corpora
python -m textblob.download_corpora

# 4. Run the app
streamlit run realtime_app.py
```

Then open your browser at **http://localhost:8501**

## 📸 How It Works
1. Type anything in the text box
2. Every 0.8 seconds the app reads your text
3. TextBlob calculates polarity & subjectivity score
4. All charts and highlights update instantly

## 📁 Project Structure
```
livesense-sentiment-analyzer/
│
├── realtime_app.py       # Main Streamlit app
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

