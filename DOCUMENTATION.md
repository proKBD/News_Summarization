# News Summarization and Analysis Project Documentation

## Project Overview
This project is a comprehensive news analysis and summarization system that processes news articles about companies, performs sentiment analysis, generates summaries, and provides Hindi audio translations. The system uses advanced NLP models and integrates multiple APIs to deliver a complete analysis solution.

## Project Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/yourusername/News_summarization.git
cd News_summarization
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

### Running the Application
1. Start the API server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8005
```

2. Start the Streamlit frontend:
```bash
streamlit run app.py
```

## Model Details

### 1. Sentiment Analysis Models
- **Primary Model**: `yiyanghkust/finbert-tone`
  - Purpose: Financial sentiment analysis
  - Capabilities: Analyzes financial news sentiment with high accuracy
  - Output: Positive, negative, or neutral sentiment with confidence scores

- **Fine-Grained Models**:
  - Financial: `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`
  - Emotion: `j-hartmann/emotion-english-distilroberta-base`
  - Aspect: `yangheng/deberta-v3-base-absa-v1.1`
  - ESG: `yiyanghkust/finbert-esg`
  - News Tone: `ProsusAI/finbert`

### 2. Summarization Model
- **Model**: `t5-base`
- **Purpose**: Generate concise summaries of news articles
- **Features**:
  - Extracts key information
  - Maintains context
  - Handles various article lengths

### 3. Text-to-Speech (TTS)
- **Model**: Google Text-to-Speech (gTTS)
- **Purpose**: Generate Hindi audio summaries
- **Features**:
  - Natural-sounding speech
  - Multiple language support
  - Adjustable speed and volume

## API Development

### API Endpoints

1. **Health Check**
```
GET /health
Purpose: Verify API server status
Response: {"status": "healthy"}
```

2. **Company Analysis**
```
POST /api/analyze
Purpose: Analyze company news
Request Body: {"name": "company_name"}
Response: {
    "articles": [...],
    "comparative_sentiment_score": {...},
    "final_sentiment_analysis": "...",
    "audio_url": "..."
}
```

3. **Audio Generation**
```
GET /audio/{filename}
Purpose: Retrieve generated audio files
Response: Audio file (MP3)
```

### API Testing with Postman

1. **Health Check**
```http
GET http://localhost:8005/health
```

2. **Company Analysis**
```http
POST http://localhost:8005/api/analyze
Content-Type: application/json

{
    "name": "Tesla"
}
```

3. **Audio Retrieval**
```http
GET http://localhost:8005/audio/Tesla_summary.mp3
```

## Third-Party API Integration

### 1. News Sources
- Google News
- Bing News
- Yahoo News
- Reuters
- MarketWatch
- Investing.com
- TechCrunch
- ZDNet

### 2. Translation Services
- Google Translate API
  - Purpose: Translate summaries to Hindi
  - Integration: Through googletrans library

### 3. Text-to-Speech
- Google Text-to-Speech (gTTS)
  - Purpose: Generate Hindi audio
  - Features: Natural language processing

## Assumptions & Limitations

### Assumptions
1. **Data Availability**
   - Assumes consistent access to news sources
   - Assumes articles contain sufficient content for analysis
   - Assumes articles are in English

2. **Model Performance**
   - Assumes models can handle various article lengths
   - Assumes sentiment analysis is accurate for financial news
   - Assumes summarization maintains key information

3. **User Input**
   - Assumes company names are provided in English
   - Assumes valid company names are entered
   - Assumes internet connectivity is available

### Limitations
1. **Data Collection**
   - Limited to 20-50 articles per analysis
   - Maximum 10 articles per source
   - May not capture all relevant news

2. **Model Constraints**
   - Sentiment analysis may be less accurate for non-financial news
   - Summarization may miss nuanced details
   - TTS quality depends on internet connectivity

3. **Technical Limitations**
   - Requires stable internet connection
   - Processing time depends on article count
   - Audio generation may have delays

4. **Language Support**
   - Primary analysis in English
   - Hindi translation available for summaries only
   - Limited to supported languages in translation

## Error Handling
- API connection issues
- Model loading failures
- Translation errors
- Audio generation failures
- Invalid company names
- Insufficient article count

## Performance Considerations
- Caching implemented for API responses
- Rate limiting for external APIs
- Batch processing for multiple articles
- Optimized model loading

## Security Considerations
- API key management
- Rate limiting
- Input validation
- Error message sanitization

## Future Improvements
1. Add more language support
2. Implement advanced caching
3. Add more news sources
4. Improve sentiment analysis accuracy
5. Enhance summarization quality
6. Add user authentication
7. Implement API key rotation
8. Add more visualization options 