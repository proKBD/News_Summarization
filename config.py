"""Configuration settings for the News Summarization application."""

# API Settings
API_HOST = "localhost"
API_PORT = 8005  # Changed from 8001 to 8005
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

# News Scraping Settings
ARTICLES_PER_SOURCE = 10  # New setting for per-source limit
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# RSS Feed Settings
RSS_FEEDS = {
    "BBC": "http://feeds.bbci.co.uk/news/business/rss.xml",
    "CNN": "http://rss.cnn.com/rss/money_news_international.rss",
    "FoxBusiness": "http://feeds.foxnews.com/foxbusiness/latest"
}

# Model Settings
SENTIMENT_MODEL = "ProsusAI/finbert"  # Financial sentiment analysis model
SUMMARIZATION_MODEL = "t5-base"

# Cache Settings
CACHE_DIR = ".cache"
CACHE_EXPIRY = 3600  # 1 hour
CACHE_DURATION = 300  # 5 minutes in seconds

# Audio Settings
AUDIO_OUTPUT_DIR = "audio_output"
DEFAULT_LANG = "hi"  # Hindi

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
