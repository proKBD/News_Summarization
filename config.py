"""Configuration settings for the News Summarization application."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8005"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")

# News Scraping Settings
ARTICLES_PER_SOURCE = int(os.getenv("ARTICLES_PER_SOURCE", "10"))
USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

# RSS Feed Settings
RSS_FEEDS = {
    "BBC": "http://feeds.bbci.co.uk/news/business/rss.xml",
    "CNN": "http://rss.cnn.com/rss/money_news_international.rss",
    "FoxBusiness": "http://feeds.foxnews.com/foxbusiness/latest"
}

# Model Settings
SENTIMENT_MODEL = "yiyanghkust/finbert-tone"  # More advanced financial sentiment model
SENTIMENT_FINE_GRAINED_MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
SUMMARIZATION_MODEL = "t5-base"

# Additional Fine-Grained Sentiment Models
FINE_GRAINED_MODELS = {
    "financial": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    "emotion": "j-hartmann/emotion-english-distilroberta-base",
    "aspect": "yangheng/deberta-v3-base-absa-v1.1",
    "esg": "yiyanghkust/finbert-esg",
    "news_tone": "ProsusAI/finbert"
}

# Fine-Grained Sentiment Categories
SENTIMENT_CATEGORIES = {
    "financial": ["positive", "negative", "neutral"],
    "emotion": ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"],
    "aspect": ["positive", "negative", "neutral"],
    "esg": ["environmental", "social", "governance", "neutral"],
    "news_tone": ["positive", "negative", "neutral"]
}

# Cache Settings
CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
CACHE_EXPIRY = int(os.getenv("CACHE_EXPIRY", "3600"))  # 1 hour
CACHE_DURATION = int(os.getenv("CACHE_DURATION", "300"))  # 5 minutes in seconds

# Audio Settings
AUDIO_OUTPUT_DIR = os.getenv("AUDIO_OUTPUT_DIR", "audio_output")
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "hi")  # Hindi

# News Sources
NEWS_SOURCES = {
    # Major News Aggregators
    "google": "https://www.google.com/search?q={}&tbm=nws",
    "bing": "https://www.bing.com/news/search?q={}",
    "yahoo": "https://news.search.yahoo.com/search?p={}",
    
    # Financial News
    "reuters": "https://www.reuters.com/search/news?blob={}",
    "marketwatch": "https://www.marketwatch.com/search?q={}&ts=0&tab=All%20News",
    "investing": "https://www.investing.com/search/?q={}&tab=news",
    
    # Tech News
    "techcrunch": "https://techcrunch.com/search/{}",
    "zdnet": "https://www.zdnet.com/search/?q={}",
}

# Article limits
MIN_ARTICLES = 20
MAX_ARTICLES_PER_SOURCE = 10  # Adjusted for more sources
MAX_ARTICLES = 50  # Increased to accommodate more sources

# Browser Headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive"
}
