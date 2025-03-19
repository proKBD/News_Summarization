# News Summarization and Text-to-Speech Application

This application extracts news articles about a company, performs sentiment analysis, and generates a Hindi text-to-speech summary.

## Features

- News article extraction using BeautifulSoup
- Sentiment analysis of articles
- Comparative sentiment analysis across articles
- Text-to-Speech conversion to Hindi
- Web interface using Streamlit
- RESTful API backend

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd News_summarization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
# Start the API server
uvicorn api:app --reload --port 8000

# In a new terminal, start the Streamlit interface
streamlit run app.py
```

## API Documentation

The application exposes the following API endpoints:

- `POST /api/analyze`: Analyze news articles for a company
- `GET /api/companies`: Get list of supported companies
- `POST /api/generate-audio`: Generate Hindi TTS for a summary

## Models Used

- Sentiment Analysis: DistilBERT fine-tuned on financial news
- Text Summarization: T5-base fine-tuned for news summarization
- TTS: gTTS (Google Text-to-Speech)

## Limitations

- Only supports non-JavaScript enabled news websites
- Limited to English language news articles for analysis
- Audio generation may take some time for longer summaries

## Project Structure

```
News_summarization/
├── app.py              # Streamlit frontend
├── api.py             # FastAPI backend
├── utils.py           # Utility functions
├── config.py          # Configuration settings
├── requirements.txt   # Project dependencies
└── README.md          # Documentation
```

## License

MIT License
