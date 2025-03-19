"""FastAPI backend for the News Summarization application."""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
from utils import NewsExtractor, SentimentAnalyzer, TextToSpeechConverter, ComparativeAnalyzer
import os
from config import API_PORT, AUDIO_OUTPUT_DIR
import time

app = FastAPI(title="News Summarization API")

# Mount static directory for audio files
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_OUTPUT_DIR), name="audio")

# Initialize components
news_extractor = NewsExtractor()
sentiment_analyzer = SentimentAnalyzer()
tts_converter = TextToSpeechConverter()
comparative_analyzer = ComparativeAnalyzer()

class CompanyRequest(BaseModel):
    name: str

class AnalysisResponse(BaseModel):
    company: str
    articles: List[Dict[str, Any]]
    comparative_sentiment_score: Dict[str, Any]
    final_sentiment_analysis: str
    audio_url: str = None

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_company(request: CompanyRequest):
    """Analyze news articles for a given company."""
    try:
        # Extract news articles
        articles = news_extractor.search_news(request.name)
        if not articles:
            raise HTTPException(status_code=404, detail="No articles found for the company")
        
        # Analyze each article
        analyzed_articles = []
        for article in articles:
            analysis = sentiment_analyzer.analyze_article(article)
            analyzed_articles.append(analysis)
        
        # Perform comparative analysis
        comparison = comparative_analyzer.analyze_coverage(analyzed_articles)
        final_analysis = comparison["final_sentiment"]
        
        # Generate Hindi audio for final analysis
        audio_filename = f"{request.name.lower().replace(' ', '_')}_{int(time.time())}"
        audio_path = tts_converter.generate_audio(final_analysis, audio_filename)
        audio_url = f"/audio/{os.path.basename(audio_path)}"
        
        return {
            "company": request.name,
            "articles": analyzed_articles,
            "comparative_sentiment_score": comparison,
            "final_sentiment_analysis": final_analysis,
            "audio_url": audio_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
