"""Utility functions for news extraction, sentiment analysis, and text-to-speech."""

import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from gtts import gTTS
import os
from typing import List, Dict, Any
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from config import *
import re
from datetime import datetime, timedelta
import time
import json
from googletrans import Translator
import statistics

class NewsExtractor:
    def __init__(self):
        self.headers = HEADERS

    def search_news(self, company_name: str) -> List[Dict[str, str]]:
        """Extract news articles about the company ensuring minimum count."""
        all_articles = []
        retries = 2  # Number of retries if we don't get enough articles
        
        while retries > 0 and len(all_articles) < MIN_ARTICLES:
            for source, url_template in NEWS_SOURCES.items():
                try:
                    url = url_template.format(company_name.replace(" ", "+"))
                    print(f"\nSearching {source} for news about {company_name}...")
                    
                    # Try different page numbers for more articles
                    for page in range(2):  # Try first two pages
                        page_url = url
                        if page > 0:
                            if source == "google":
                                page_url += f"&start={page * 10}"
                            elif source == "bing":
                                page_url += f"&first={page * 10 + 1}"
                            elif source == "yahoo":
                                page_url += f"&b={page * 10 + 1}"
                            elif source == "reuters":
                                page_url += f"&page={page + 1}"
                            elif source == "marketwatch":
                                page_url += f"&page={page + 1}"
                            elif source == "investing":
                                page_url += f"&page={page + 1}"
                            elif source == "techcrunch":
                                page_url += f"/page/{page + 1}"
                            elif source == "zdnet":
                                page_url += f"&page={page + 1}"
                        
                        response = requests.get(page_url, headers=self.headers, timeout=15)
                        if response.status_code != 200:
                            print(f"Error: {source} page {page+1} returned status code {response.status_code}")
                            continue
                            
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        source_articles = []
                        if source == "google":
                            source_articles = self._parse_google_news(soup)
                        elif source == "bing":
                            source_articles = self._parse_bing_news(soup)
                        elif source == "yahoo":
                            source_articles = self._parse_yahoo_news(soup)
                        elif source == "reuters":
                            source_articles = self._parse_reuters_news(soup)
                        elif source == "marketwatch":
                            source_articles = self._parse_marketwatch_news(soup)
                        elif source == "investing":
                            source_articles = self._parse_investing_news(soup)
                        elif source == "techcrunch":
                            source_articles = self._parse_techcrunch_news(soup)
                        elif source == "zdnet":
                            source_articles = self._parse_zdnet_news(soup)
                        
                        # Limit articles per source
                        if source_articles:
                            source_articles = source_articles[:MAX_ARTICLES_PER_SOURCE]
                            all_articles.extend(source_articles)
                            print(f"Found {len(source_articles)} articles from {source} page {page+1}")
                        
                        # If we have enough articles, break the page loop
                        if len(all_articles) >= MIN_ARTICLES:
                            break
                            
                except Exception as e:
                    print(f"Error fetching from {source}: {str(e)}")
                    continue
                
                # If we have enough articles, break the source loop
                if len(all_articles) >= MIN_ARTICLES:
                    break
            
            retries -= 1
            if len(all_articles) < MIN_ARTICLES and retries > 0:
                print(f"\nFound only {len(all_articles)} articles, retrying...")
        
        # Remove duplicates
        unique_articles = self._remove_duplicates(all_articles)
        print(f"\nFound {len(unique_articles)} unique articles")
        
        if len(unique_articles) < MIN_ARTICLES:
            print(f"Warning: Could only find {len(unique_articles)} unique articles, fewer than minimum {MIN_ARTICLES}")
        
        # Balance articles across sources
        balanced_articles = self._balance_sources(unique_articles)
        return balanced_articles[:max(MIN_ARTICLES, MAX_ARTICLES)]

    def _balance_sources(self, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Balance articles across sources while maintaining minimum count."""
        source_articles = {}
        
        # Group articles by source
        for article in articles:
            source = article['source']
            if source not in source_articles:
                source_articles[source] = []
            source_articles[source].append(article)
        
        # Calculate target articles per source
        total_sources = len(source_articles)
        target_per_source = max(MIN_ARTICLES // total_sources, 
                              MAX_ARTICLES_PER_SOURCE)
        
        # Get articles from each source
        balanced = []
        for source, articles_list in source_articles.items():
            balanced.extend(articles_list[:target_per_source])
        
        # If we still need more articles to meet minimum, add more from sources
        # that have additional articles
        if len(balanced) < MIN_ARTICLES:
            remaining = []
            for articles_list in source_articles.values():
                remaining.extend(articles_list[target_per_source:])
            
            # Sort remaining by source to maintain balance
            remaining.sort(key=lambda x: len([a for a in balanced if a['source'] == x['source']]))
            
            while len(balanced) < MIN_ARTICLES and remaining:
                balanced.append(remaining.pop(0))
        
        return balanced

    def _parse_google_news(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Parse Google News search results."""
        articles = []
        for div in soup.find_all(['div', 'article'], class_=['g', 'xuvV6b', 'WlydOe']):
            try:
                title_elem = div.find(['h3', 'h4'])
                snippet_elem = div.find('div', class_=['VwiC3b', 'yy6M1d'])
                link_elem = div.find('a')
                source_elem = div.find(['div', 'span'], class_='UPmit')
                
                if title_elem and snippet_elem and link_elem:
                    source = source_elem.get_text(strip=True) if source_elem else 'Google News'
                    articles.append({
                        'title': title_elem.get_text(strip=True),
                        'content': snippet_elem.get_text(strip=True),
                        'url': link_elem['href'],
                        'source': source
                    })
            except Exception as e:
                print(f"Error parsing Google article: {str(e)}")
                continue
        return articles

    def _parse_bing_news(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Parse Bing News search results."""
        articles = []
        for article in soup.find_all(['div', 'article'], class_=['news-card', 'newsitem', 'item-info']):
            try:
                title_elem = article.find(['a', 'h3'], class_=['title', 'news-card-title'])
                snippet_elem = article.find(['div', 'p'], class_=['snippet', 'description'])
                source_elem = article.find(['div', 'span'], class_=['source', 'provider'])
                
                if title_elem and snippet_elem:
                    source = source_elem.get_text(strip=True) if source_elem else 'Bing News'
                    url = title_elem['href'] if 'href' in title_elem.attrs else ''
                    articles.append({
                        'title': title_elem.get_text(strip=True),
                        'content': snippet_elem.get_text(strip=True),
                        'url': url,
                        'source': source
                    })
            except Exception as e:
                print(f"Error parsing Bing article: {str(e)}")
        return articles

    def _parse_yahoo_news(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Parse Yahoo News search results."""
        articles = []
        for article in soup.find_all('div', class_='NewsArticle'):
            try:
                title_elem = article.find(['h4', 'h3', 'a'])
                snippet_elem = article.find('p')
                source_elem = article.find(['span', 'div'], class_=['provider', 'source'])
                
                if title_elem and snippet_elem:
                    source = source_elem.get_text(strip=True) if source_elem else 'Yahoo News'
                    url = title_elem.find('a')['href'] if title_elem.find('a') else ''
                    articles.append({
                        'title': title_elem.get_text(strip=True),
                        'content': snippet_elem.get_text(strip=True),
                        'url': url,
                        'source': source
                    })
            except Exception as e:
                print(f"Error parsing Yahoo article: {str(e)}")
        return articles

    def _parse_reuters_news(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Parse Reuters search results."""
        articles = []
        for article in soup.find_all(['div', 'article'], class_=['search-result-content', 'story']):
            try:
                title_elem = article.find(['h3', 'a'], class_='story-title')
                snippet_elem = article.find(['p', 'div'], class_=['story-description', 'description'])
                
                if title_elem:
                    url = title_elem.find('a')['href'] if title_elem.find('a') else ''
                    if url and not url.startswith('http'):
                        url = 'https://www.reuters.com' + url
                    
                    articles.append({
                        'title': title_elem.get_text(strip=True),
                        'content': snippet_elem.get_text(strip=True) if snippet_elem else '',
                        'url': url,
                        'source': 'Reuters'
                    })
            except Exception as e:
                print(f"Error parsing Reuters article: {str(e)}")
        return articles

    def _parse_marketwatch_news(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Parse MarketWatch search results."""
        articles = []
        for article in soup.find_all(['div', 'article'], class_=['element--article', 'article__content']):
            try:
                title_elem = article.find(['h3', 'h2'], class_=['article__headline', 'title'])
                snippet_elem = article.find('p', class_=['article__summary', 'description'])
                
                if title_elem:
                    url = title_elem.find('a')['href'] if title_elem.find('a') else ''
                    articles.append({
                        'title': title_elem.get_text(strip=True),
                        'content': snippet_elem.get_text(strip=True) if snippet_elem else '',
                        'url': url,
                        'source': 'MarketWatch'
                    })
            except Exception as e:
                print(f"Error parsing MarketWatch article: {str(e)}")
        return articles

    def _parse_investing_news(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Parse Investing.com search results."""
        articles = []
        for article in soup.find_all(['div', 'article'], class_=['articleItem', 'news-item']):
            try:
                title_elem = article.find(['a', 'h3'], class_=['title', 'articleTitle'])
                snippet_elem = article.find(['p', 'div'], class_=['description', 'articleContent'])
                
                if title_elem:
                    url = title_elem['href'] if 'href' in title_elem.attrs else title_elem.find('a')['href']
                    if url and not url.startswith('http'):
                        url = 'https://www.investing.com' + url
                        
                    articles.append({
                        'title': title_elem.get_text(strip=True),
                        'content': snippet_elem.get_text(strip=True) if snippet_elem else '',
                        'url': url,
                        'source': 'Investing.com'
                    })
            except Exception as e:
                print(f"Error parsing Investing.com article: {str(e)}")
        return articles

    def _parse_techcrunch_news(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Parse TechCrunch search results."""
        articles = []
        for article in soup.find_all(['div', 'article'], class_=['post-block', 'article-block']):
            try:
                title_elem = article.find(['h2', 'h3', 'a'], class_=['post-block__title', 'article-title'])
                snippet_elem = article.find(['div', 'p'], class_=['post-block__content', 'article-content'])
                
                if title_elem:
                    url = title_elem.find('a')['href'] if title_elem.find('a') else ''
                    articles.append({
                        'title': title_elem.get_text(strip=True),
                        'content': snippet_elem.get_text(strip=True) if snippet_elem else '',
                        'url': url,
                        'source': 'TechCrunch'
                    })
            except Exception as e:
                print(f"Error parsing TechCrunch article: {str(e)}")
        return articles

    def _parse_zdnet_news(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Parse ZDNet search results."""
        articles = []
        for article in soup.find_all(['div', 'article'], class_=['item', 'article']):
            try:
                title_elem = article.find(['h3', 'a'], class_=['title', 'headline'])
                snippet_elem = article.find(['p', 'div'], class_=['summary', 'content'])
                
                if title_elem:
                    url = title_elem.find('a')['href'] if title_elem.find('a') else ''
                    if url and not url.startswith('http'):
                        url = 'https://www.zdnet.com' + url
                        
                    articles.append({
                        'title': title_elem.get_text(strip=True),
                        'content': snippet_elem.get_text(strip=True) if snippet_elem else '',
                        'url': url,
                        'source': 'ZDNet'
                    })
            except Exception as e:
                print(f"Error parsing ZDNet article: {str(e)}")
        return articles

    def _remove_duplicates(self, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate articles based on title similarity."""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title = article['title'].lower()
            if not any(title in seen_title or seen_title in title for seen_title in seen_titles):
                unique_articles.append(article)
                seen_titles.add(title)
        
        return unique_articles

class SentimentAnalyzer:
    def __init__(self):
        try:
            # Primary financial sentiment model
            self.sentiment_pipeline = pipeline("sentiment-analysis", 
                                      model=SENTIMENT_MODEL)
            
            # Initialize fine-grained sentiment models
            self.fine_grained_models = {}
            try:
                # Initialize the default fine-grained model for backward compatibility
                self.fine_grained_sentiment = pipeline("sentiment-analysis",
                                               model=SENTIMENT_FINE_GRAINED_MODEL)
                
                # Initialize additional fine-grained models
                for model_name, model_path in FINE_GRAINED_MODELS.items():
                    try:
                        print(f"Loading fine-grained model: {model_name}")
                        self.fine_grained_models[model_name] = pipeline("sentiment-analysis", 
                                                                model=model_path)
                    except Exception as e:
                        print(f"Error loading fine-grained model {model_name}: {str(e)}")
            except Exception as e:
                print(f"Error initializing fine-grained models: {str(e)}")
                self.fine_grained_sentiment = None
            
            # Initialize additional sentiment analyzers if available
            self.has_textblob = False
            self.has_vader = False
            
            try:
                from textblob import TextBlob
                self.TextBlob = TextBlob
                self.has_textblob = True
            except:
                print("TextBlob not available. Install with: pip install textblob")
            
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.vader = SentimentIntensityAnalyzer()
                self.has_vader = True
            except:
                print("VADER not available. Install with: pip install vaderSentiment")
            
            self.summarizer = pipeline("summarization", 
                               model=SUMMARIZATION_MODEL)
            self.vectorizer = TfidfVectorizer(stop_words='english', 
                                      max_features=10)
            
            # Initialize NER pipeline if spaCy is available
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                self.has_ner = True
            except:
                self.has_ner = False
                print("spaCy not available for NER. Install with: pip install spacy && python -m spacy download en_core_web_sm")
                
        except Exception as e:
            print(f"Error initializing sentiment models: {str(e)}")
            # Fallback to default models if specific models fail
            self.sentiment_pipeline = pipeline("sentiment-analysis")
            self.fine_grained_sentiment = None
            self.fine_grained_models = {}
            self.summarizer = pipeline("summarization")
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
            self.has_ner = False
            self.has_textblob = False
            self.has_vader = False

    def analyze_article(self, article: Dict[str, str]) -> Dict[str, Any]:
        """Analyze sentiment and generate summary for an article."""
        try:
            # Get the full text by combining title and content
            full_text = f"{article['title']} {article['content']}"
            
            # Generate summary
            summary = self.summarize_text(full_text)
            
            # Get ensemble sentiment analysis
            sentiment_analysis = self._get_ensemble_sentiment(full_text)
            sentiment_label = sentiment_analysis['ensemble_sentiment']
            sentiment_score = sentiment_analysis['ensemble_score']
            
            # Add fine-grained sentiment analysis
            fine_grained_sentiment = self._get_fine_grained_sentiment(full_text)
            
            # Extract key topics
            topics = self.extract_topics(full_text)
            
            # Extract named entities
            entities = self._extract_entities(full_text)
            
            # Extract sentiment targets (entities associated with sentiment)
            sentiment_targets = self._extract_sentiment_targets(full_text, entities)
            
            # Add analysis to article
            analyzed_article = article.copy()
            analyzed_article.update({
                'summary': summary,
                'sentiment': sentiment_label,
                'sentiment_score': sentiment_score,
                'sentiment_details': sentiment_analysis,
                'fine_grained_sentiment': fine_grained_sentiment,
                'topics': topics,
                'entities': entities,
                'sentiment_targets': sentiment_targets,
                'sentiment_indices': fine_grained_sentiment.get('indices', {}),
                'analysis_timestamp': datetime.now().isoformat()
            })
            
            return analyzed_article
            
        except Exception as e:
            print(f"Error analyzing article: {str(e)}")
            # Return original article with default values if analysis fails
            article.update({
                'summary': article.get('content', '')[:200] + '...',
                'sentiment': 'neutral',
                'sentiment_score': 0.0,
                'sentiment_details': {},
                'fine_grained_sentiment': {},
                'topics': [],
                'entities': {},
                'sentiment_targets': [],
                'sentiment_indices': {
                    'positivity_index': 0.5,
                    'negativity_index': 0.5,
                    'emotional_intensity': 0.0,
                    'controversy_score': 0.0,
                    'confidence_score': 0.0,
                    'esg_relevance': 0.0
                },
                'analysis_timestamp': datetime.now().isoformat()
            })
            return article

    def _get_ensemble_sentiment(self, text: str) -> Dict[str, Any]:
        """Get ensemble sentiment by combining multiple sentiment models."""
        results = {}
        
        # Initialize with default values
        ensemble_result = {
            'ensemble_sentiment': 'neutral',
            'ensemble_score': 0.5,
            'models': {}
        }
        
        try:
            # 1. Primary transformer model (finbert)
            try:
                primary_result = self.sentiment_pipeline(text[:512])[0]  # Limit text length
                primary_label = primary_result['label'].lower()
                primary_score = primary_result['score']
                
                # Map to standard format
                if primary_label == 'positive':
                    primary_normalized = primary_score
                elif primary_label == 'negative':
                    primary_normalized = 1 - primary_score
                else:  # neutral
                    primary_normalized = 0.5
                    
                ensemble_result['models']['transformer'] = {
                    'sentiment': primary_label,
                    'score': round(primary_score, 3),
                    'normalized_score': round(primary_normalized, 3)
                }
            except:
                ensemble_result['models']['transformer'] = {
                    'sentiment': 'error',
                    'score': 0,
                    'normalized_score': 0.5
                }
            
            # 2. TextBlob sentiment
            if self.has_textblob:
                try:
                    blob = self.TextBlob(text)
                    polarity = blob.sentiment.polarity
                    
                    # Convert to standard format
                    if polarity > 0.1:
                        textblob_sentiment = 'positive'
                        textblob_score = polarity
                    elif polarity < -0.1:
                        textblob_sentiment = 'negative'
                        textblob_score = abs(polarity)
                    else:
                        textblob_sentiment = 'neutral'
                        textblob_score = 0.5
                        
                    # Normalize to 0-1 scale
                    textblob_normalized = (polarity + 1) / 2
                    
                    ensemble_result['models']['textblob'] = {
                        'sentiment': textblob_sentiment,
                        'score': round(textblob_score, 3),
                        'normalized_score': round(textblob_normalized, 3)
                    }
                except:
                    ensemble_result['models']['textblob'] = {
                        'sentiment': 'error',
                        'score': 0,
                        'normalized_score': 0.5
                    }
            
            # 3. VADER sentiment
            if self.has_vader:
                try:
                    vader_scores = self.vader.polarity_scores(text)
                    compound = vader_scores['compound']
                    
                    # Convert to standard format
                    if compound > 0.05:
                        vader_sentiment = 'positive'
                        vader_score = compound
                    elif compound < -0.05:
                        vader_sentiment = 'negative'
                        vader_score = abs(compound)
                    else:
                        vader_sentiment = 'neutral'
                        vader_score = 0.5
                        
                    # Normalize to 0-1 scale
                    vader_normalized = (compound + 1) / 2
                    
                    ensemble_result['models']['vader'] = {
                        'sentiment': vader_sentiment,
                        'score': round(vader_score, 3),
                        'normalized_score': round(vader_normalized, 3)
                    }
                except:
                    ensemble_result['models']['vader'] = {
                        'sentiment': 'error',
                        'score': 0,
                        'normalized_score': 0.5
                    }
            
            # Calculate ensemble result
            # Get all normalized scores
            normalized_scores = []
            for model_name, model_result in ensemble_result['models'].items():
                if model_result['sentiment'] != 'error':
                    normalized_scores.append(model_result['normalized_score'])
            
            # Calculate average if we have scores
            if normalized_scores:
                avg_score = sum(normalized_scores) / len(normalized_scores)
                
                # Convert to sentiment label
                if avg_score > 0.6:
                    ensemble_sentiment = 'positive'
                elif avg_score < 0.4:
                    ensemble_sentiment = 'negative'
                else:
                    ensemble_sentiment = 'neutral'
                    
                ensemble_result['ensemble_sentiment'] = ensemble_sentiment
                ensemble_result['ensemble_score'] = round(avg_score, 3)
            
            # Add confidence level
            if len(normalized_scores) > 1:
                # Calculate standard deviation to measure agreement
                std_dev = statistics.stdev(normalized_scores) if len(normalized_scores) > 1 else 0
                agreement = 1 - (std_dev * 2)  # Lower std_dev means higher agreement
                agreement = max(0, min(1, agreement))  # Clamp to 0-1
                
                ensemble_result['model_agreement'] = round(agreement, 3)
            
            return ensemble_result
            
        except Exception as e:
            print(f"Error in ensemble sentiment analysis: {str(e)}")
            return {
                'ensemble_sentiment': 'neutral',
                'ensemble_score': 0.5,
                'models': {}
            }

    def _get_fine_grained_sentiment(self, text: str) -> Dict[str, Any]:
        """Get fine-grained sentiment analysis with more detailed categories."""
        # Initialize result structure
        result = {
            "primary": {"category": "unknown", "confidence": 0.0},
            "models": {}
        }
        
        # Check if we have any fine-grained models
        if not self.fine_grained_sentiment and not self.fine_grained_models:
            return result
            
        try:
            # Split text into manageable chunks if too long
            chunks = self._split_text(text)
            
            # Process with default fine-grained model for backward compatibility
            if self.fine_grained_sentiment:
                primary_results = []
                
                for chunk in chunks:
                    if not chunk.strip():
                        continue
                    chunk_result = self.fine_grained_sentiment(chunk)[0]
                    primary_results.append(chunk_result)
                
                if primary_results:
                    # Aggregate results from all chunks
                    categories = {}
                    for res in primary_results:
                        label = res['label'].lower()
                        score = res['score']
                        if label in categories:
                            categories[label] += score
                        else:
                            categories[label] = score
                    
                    # Normalize scores
                    total = sum(categories.values())
                    if total > 0:
                        categories = {k: round(v/total, 3) for k, v in categories.items()}
                    
                    # Get dominant category
                    dominant_category = max(categories.items(), key=lambda x: x[1])
                    
                    result["primary"] = {
                        "category": dominant_category[0],
                        "confidence": dominant_category[1],
                        "distribution": categories
                    }
            
            # Process with additional fine-grained models
            for model_name, model in self.fine_grained_models.items():
                model_results = []
                
                for chunk in chunks:
                    if not chunk.strip():
                        continue
                    try:
                        chunk_result = model(chunk)[0]
                        model_results.append(chunk_result)
                    except Exception as e:
                        print(f"Error analyzing chunk with model {model_name}: {str(e)}")
                
                if model_results:
                    # Aggregate results from all chunks
                    categories = {}
                    for res in model_results:
                        # Ensure the label is lowercase for consistency
                        label = res['label'].lower() if isinstance(res.get('label'), str) else "unknown"
                        score = res['score']
                        if label in categories:
                            categories[label] += score
                        else:
                            categories[label] = score
                    
                    # Normalize scores
                    total = sum(categories.values())
                    if total > 0:
                        categories = {k: round(v/total, 3) for k, v in categories.items()}
                    
                    # Get dominant category
                    dominant_category = max(categories.items(), key=lambda x: x[1])
                    
                    # Store results for this model
                    result["models"][model_name] = {
                        "category": dominant_category[0],
                        "confidence": dominant_category[1],
                        "distribution": categories
                    }
            
            # Calculate sentiment indices based on the fine-grained results
            result["indices"] = self._calculate_sentiment_indices(result)
            
            return result
            
        except Exception as e:
            print(f"Error in fine-grained sentiment analysis: {str(e)}")
            return result
    
    def _calculate_sentiment_indices(self, fine_grained_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various sentiment indices based on fine-grained sentiment analysis."""
        indices = {
            "positivity_index": 0.5,  # Default neutral value
            "negativity_index": 0.5,
            "emotional_intensity": 0.0,
            "controversy_score": 0.0,
            "confidence_score": 0.0,
            "esg_relevance": 0.0
        }
        
        try:
            # Extract distributions from all models
            distributions = {}
            confidence_scores = {}
            
            # Add primary model if available
            if "category" in fine_grained_results.get("primary", {}):
                if "distribution" in fine_grained_results["primary"]:
                    distributions["primary"] = fine_grained_results["primary"]["distribution"]
                confidence_scores["primary"] = fine_grained_results["primary"].get("confidence", 0.0)
            
            # Add other models
            for model_name, model_result in fine_grained_results.get("models", {}).items():
                if "distribution" in model_result:
                    distributions[model_name] = model_result["distribution"]
                confidence_scores[model_name] = model_result.get("confidence", 0.0)
            
            # Calculate positivity index
            positive_scores = []
            for model_name, dist in distributions.items():
                if model_name == "financial" or model_name == "primary" or model_name == "news_tone" or model_name == "aspect":
                    pos_score = dist.get("positive", 0.0)
                    positive_scores.append(pos_score)
                elif model_name == "emotion":
                    # For emotion model, consider joy as positive
                    pos_score = dist.get("joy", 0.0) + dist.get("surprise", 0.0) * 0.5
                    positive_scores.append(pos_score)
            
            if positive_scores:
                indices["positivity_index"] = round(sum(positive_scores) / len(positive_scores), 3)
            
            # Calculate negativity index
            negative_scores = []
            for model_name, dist in distributions.items():
                if model_name == "financial" or model_name == "primary" or model_name == "news_tone" or model_name == "aspect":
                    neg_score = dist.get("negative", 0.0)
                    negative_scores.append(neg_score)
                elif model_name == "emotion":
                    # For emotion model, consider sadness, anger, fear, disgust as negative
                    neg_score = dist.get("sadness", 0.0) + dist.get("anger", 0.0) + \
                                dist.get("fear", 0.0) + dist.get("disgust", 0.0)
                    negative_scores.append(neg_score / 4)  # Average of 4 negative emotions
            
            if negative_scores:
                indices["negativity_index"] = round(sum(negative_scores) / len(negative_scores), 3)
            
            # Calculate emotional intensity
            emotion_dist = distributions.get("emotion", {})
            if emotion_dist:
                # Sum all emotional intensities except neutral
                emotional_sum = sum(v for k, v in emotion_dist.items() if k != "neutral")
                indices["emotional_intensity"] = round(emotional_sum, 3)
            
            # Calculate controversy score (high when both positive and negative are high)
            indices["controversy_score"] = round(indices["positivity_index"] * indices["negativity_index"] * 4, 3)
            
            # Calculate confidence score (average of all model confidences)
            if confidence_scores:
                indices["confidence_score"] = round(sum(confidence_scores.values()) / len(confidence_scores), 3)
            
            # Calculate ESG relevance if available
            esg_dist = distributions.get("esg", {})
            if esg_dist:
                # Sum of all ESG categories
                esg_sum = sum(v for k, v in esg_dist.items() if k in ["environmental", "social", "governance"])
                indices["esg_relevance"] = round(esg_sum, 3)
            
            return indices
            
        except Exception as e:
            print(f"Error calculating sentiment indices: {str(e)}")
            return indices

    def summarize_text(self, text: str) -> str:
        """Generate a concise summary of the text."""
        try:
            # Clean and prepare text
            text = text.replace('\n', ' ').strip()
            
            # Split text into chunks if it's too long
            chunks = self._split_text(text)
            
            summaries = []
            for chunk in chunks:
                # Generate summary for each chunk
                summary = self.summarizer(chunk, 
                                       max_length=130, 
                                       min_length=30, 
                                       do_sample=False)[0]['summary_text']
                summaries.append(summary)
            
            # Combine summaries if there were multiple chunks
            final_summary = ' '.join(summaries)
            return final_summary
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return text[:200] + '...'  # Return truncated text as fallback

    def extract_topics(self, text: str) -> List[str]:
        """Extract key topics from the text using TF-IDF."""
        try:
            # Prepare text
            text = text.lower()
            
            # Fit and transform the text
            tfidf_matrix = self.vectorizer.fit_transform([text])
            
            # Get feature names and scores
            feature_names = self.vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top topics
            top_indices = scores.argsort()[-5:][::-1]  # Get top 5 topics
            topics = [feature_names[i] for i in top_indices]
            
            return topics
            
        except Exception as e:
            print(f"Error extracting topics: {str(e)}")
            return []

    def _split_text(self, text: str, max_length: int = 1024) -> List[str]:
        """Split text into chunks that fit within model's maximum token limit."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Countries, cities, states
            'MONEY': [],
            'PERCENT': [],
            'DATE': []
        }
        
        if not self.has_ner:
            return entities
            
        try:
            # Process text with spaCy
            doc = self.nlp(text[:10000])  # Limit text length for performance
            
            # Extract entities
            for ent in doc.ents:
                if ent.label_ in entities:
                    # Clean entity text and deduplicate
                    clean_text = ent.text.strip()
                    if clean_text and clean_text not in entities[ent.label_]:
                        entities[ent.label_].append(clean_text)
            
            return entities
        except Exception as e:
            print(f"Error extracting entities: {str(e)}")
            return entities
    
    def _extract_sentiment_targets(self, text: str, entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Extract entities that are targets of sentiment expressions."""
        if not self.has_ner:
            return []
            
        try:
            # Get all entities as a flat list
            all_entities = []
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    all_entities.append({
                        'text': entity,
                        'type': entity_type
                    })
            
            # Find sentiment targets
            targets = []
            
            # Split text into sentences
            doc = self.nlp(text[:10000])  # Limit text length
            
            for sentence in doc.sents:
                # Skip short sentences
                if len(sentence.text.split()) < 3:
                    continue
                    
                # Check for sentiment in this sentence
                try:
                    sentiment = self.sentiment_pipeline(sentence.text)[0]
                    # Only process if sentiment is strong
                    if sentiment['score'] > 0.7:
                        # Find entities in this sentence
                        for entity in all_entities:
                            if entity['text'] in sentence.text:
                                targets.append({
                                    'entity': entity['text'],
                                    'type': entity['type'],
                                    'sentiment': sentiment['label'].lower(),
                                    'confidence': round(sentiment['score'], 3),
                                    'context': sentence.text
                                })
                except:
                    continue
            
            # Return unique targets
            unique_targets = []
            seen = set()
            for target in targets:
                key = f"{target['entity']}_{target['sentiment']}"
                if key not in seen:
                    seen.add(key)
                    unique_targets.append(target)
            
            return unique_targets
            
        except Exception as e:
            print(f"Error extracting sentiment targets: {str(e)}")
            return []

class TextToSpeechConverter:
    def __init__(self):
        self.output_dir = AUDIO_OUTPUT_DIR
        self.translator = Translator()
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_audio(self, text: str, filename: str) -> str:
        """Convert text to Hindi speech and save as audio file."""
        try:
            print(f"Translating text to Hindi: {text[:100]}...")
            
            # First translate the text to Hindi
            # Use chunking for long text to avoid translation limits
            chunks = []
            for i in range(0, len(text), 1000):
                chunk = text[i:i+1000]
                try:
                    translated_chunk = self.translator.translate(chunk, dest='hi').text
                    chunks.append(translated_chunk)
                    print(f"Translated chunk {i//1000 + 1}")
                except Exception as e:
                    print(f"Error translating chunk {i//1000 + 1}: {str(e)}")
                    # If translation fails, use original text
                    chunks.append(chunk)
            
            hindi_text = ' '.join(chunks)
            print(f"Translation complete. Hindi text length: {len(hindi_text)}")
            
            # Generate Hindi speech
            print("Generating Hindi speech...")
            tts = gTTS(text=hindi_text, lang='hi', slow=False)
            output_path = os.path.join(self.output_dir, f"{filename}.mp3")
            tts.save(output_path)
            print(f"Audio saved to {output_path}")
            
            return output_path
        except Exception as e:
            print(f"Error in TTS conversion: {str(e)}")
            # Fallback to original text if translation fails
            print("Using fallback English TTS")
            tts = gTTS(text=text, lang='en')
            output_path = os.path.join(self.output_dir, f"{filename}.mp3")
            tts.save(output_path)
            return output_path

class ComparativeAnalyzer:
    def __init__(self):
        pass
    
    def analyze_coverage(self, articles: List[Dict[str, Any]], company_name: str = None) -> Dict[str, Any]:
        """Perform comparative analysis across articles."""
        if not articles:
            return {
                "topics": [],
                "sentiment_distribution": {},
                "coverage_differences": ["No articles found for analysis."],
                "final_sentiment": "No articles found for analysis.",
                "total_articles": 0,
                "sentiment_indices": {}
            }
        
        # Add company name to each article if provided
        if company_name:
            for article in articles:
                article['company'] = company_name
        
        # Calculate sentiment distribution
        sentiment_dist = self._get_sentiment_distribution(articles)
        
        # Analyze common topics
        topics = self._analyze_topics(articles)
        
        # Analyze coverage differences
        differences = self._analyze_coverage_differences(articles)
        
        # Get final sentiment analysis
        final_sentiment = self._get_final_sentiment(sentiment_dist, articles)
        
        return {
            "topics": topics,
            "sentiment_distribution": sentiment_dist,
            "coverage_differences": differences,
            "final_sentiment": final_sentiment,
            "total_articles": len(articles),
            "sentiment_indices": sentiment_dist.get("sentiment_indices", {})
        }

    def _get_sentiment_distribution(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate distribution of sentiments across articles."""
        # Basic sentiment distribution
        basic_distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        # Fine-grained sentiment distribution
        fine_grained_distribution = {}
        
        # Sentiment scores
        sentiment_scores = []
        
        # Sentiment indices aggregation
        sentiment_indices = {
            "positivity_index": [],
            "negativity_index": [],
            "emotional_intensity": [],
            "controversy_score": [],
            "confidence_score": [],
            "esg_relevance": []
        }
        
        # Process each article
        for article in articles:
            # Basic sentiment
            sentiment = article.get('sentiment', 'neutral').lower()
            if isinstance(sentiment, str):
                basic_distribution[sentiment] = basic_distribution.get(sentiment, 0) + 1
            
            # Sentiment score
            score = article.get('sentiment_score', 0.0)
            sentiment_scores.append(score)
            
            # Fine-grained sentiment
            fine_grained = article.get('fine_grained_sentiment', {})
            if fine_grained and 'category' in fine_grained:
                category = fine_grained['category'].lower() if isinstance(fine_grained['category'], str) else "unknown"
                fine_grained_distribution[category] = fine_grained_distribution.get(category, 0) + 1
            
            # Collect sentiment indices
            indices = article.get('sentiment_indices', {})
            for index_name, index_values in sentiment_indices.items():
                if index_name in indices:
                    index_values.append(indices[index_name])
        
        # Calculate average sentiment score
        avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # Calculate sentiment volatility (standard deviation)
        sentiment_volatility = 0
        if len(sentiment_scores) > 1:
            sentiment_volatility = statistics.stdev(sentiment_scores)
        
        # Calculate average sentiment indices
        avg_indices = {}
        for index_name, values in sentiment_indices.items():
            if values:
                avg_indices[index_name] = round(sum(values) / len(values), 3)
            else:
                avg_indices[index_name] = 0.0
        
        return {
            "basic": basic_distribution,
            "fine_grained": fine_grained_distribution,
            "avg_score": round(avg_sentiment_score, 3),
            "volatility": round(sentiment_volatility, 3),
            "sentiment_indices": avg_indices
        }
    
    def _analyze_topics(self, articles: List[Dict[str, Any]]) -> List[str]:
        """Analyze common topics across articles using TF-IDF."""
        try:
            # Combine title and content for better topic extraction
            texts = [f"{article.get('title', '')} {article.get('content', '')}" for article in articles]
            
            # Create and fit TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=10,
                stop_words='english',
                ngram_range=(1, 2),
                token_pattern=r'(?u)\b[A-Za-z][A-Za-z+\'-]*[A-Za-z]+\b'  # Improved pattern
            )
            
            # Clean and normalize texts
            cleaned_texts = []
            for text in texts:
                # Remove numbers and special characters
                cleaned = re.sub(r'\d+', '', text)
                cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
                cleaned_texts.append(cleaned.lower())
            
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores for each term
            avg_scores = tfidf_matrix.mean(axis=0).A1
            
            # Sort terms by score and return top meaningful terms
            sorted_indices = avg_scores.argsort()[-5:][::-1]
            meaningful_topics = []
            
            for idx in sorted_indices:
                topic = feature_names[idx]
                # Filter out single characters and common words
                if len(topic) > 1 and topic not in {'000', 'com', 'said', 'says', 'year', 'new', 'one'}:
                    meaningful_topics.append(topic)
                if len(meaningful_topics) >= 5:
                    break
            
            return meaningful_topics
            
        except Exception as e:
            print(f"Error analyzing topics: {str(e)}")
            return []

    def _analyze_coverage_differences(self, articles: List[Dict[str, Any]]) -> List[str]:
        """Analyze how coverage differs across articles."""
        if not articles:
            return ["No articles available for comparison"]
        
        differences = []
        
        # Compare sentiment differences
        sentiments = [article.get('sentiment', 'neutral').lower() for article in articles]
        unique_sentiments = set(sentiments)
        if len(unique_sentiments) > 1:
            pos_count = sentiments.count('positive')
            neg_count = sentiments.count('negative')
            neu_count = sentiments.count('neutral')
            
            if pos_count > 0 and neg_count > 0:
                differences.append(f"Coverage sentiment varies significantly: {pos_count} positive, {neg_count} negative, and {neu_count} neutral articles.")
        
        # Compare fine-grained sentiment differences
        fine_grained_categories = []
        for article in articles:
            fine_grained = article.get('fine_grained_sentiment', {})
            if isinstance(fine_grained, dict) and 'category' in fine_grained:
                category = fine_grained['category']
                if isinstance(category, str):
                    fine_grained_categories.append(category.lower())
        
        unique_categories = set(fine_grained_categories)
        if len(unique_categories) > 2:  # More than 2 different categories
            category_counts = {}
            for category in fine_grained_categories:
                category_counts[category] = category_counts.get(category, 0) + 1
            
            top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            categories_str = ", ".join([f"{cat} ({count})" for cat, count in top_categories])
            differences.append(f"Articles show diverse sentiment categories: {categories_str}")
        
        # Compare sentiment indices
        indices_differences = []
        positivity_values = []
        negativity_values = []
        controversy_values = []
        
        for article in articles:
            indices = article.get('sentiment_indices', {})
            if indices:
                if 'positivity_index' in indices:
                    positivity_values.append(indices['positivity_index'])
                if 'negativity_index' in indices:
                    negativity_values.append(indices['negativity_index'])
                if 'controversy_score' in indices:
                    controversy_values.append(indices['controversy_score'])
        
        # Check for high variance in positivity
        if positivity_values and len(positivity_values) > 1:
            if max(positivity_values) - min(positivity_values) > 0.4:
                indices_differences.append("Articles show significant variation in positivity levels")
        
        # Check for high variance in negativity
        if negativity_values and len(negativity_values) > 1:
            if max(negativity_values) - min(negativity_values) > 0.4:
                indices_differences.append("Articles show significant variation in negativity levels")
        
        # Check for high controversy scores
        if controversy_values:
            high_controversy = [v for v in controversy_values if v > 0.5]
            if high_controversy:
                indices_differences.append(f"{len(high_controversy)} articles show high controversy scores")
        
        if indices_differences:
            differences.append("Sentiment index analysis: " + "; ".join(indices_differences))
        
        # Compare source differences
        sources = [article.get('source', '').lower() for article in articles]
        source_counts = {}
        for source in sources:
            if source:
                source_counts[source] = source_counts.get(source, 0) + 1
        
        if len(source_counts) > 1:
            top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            sources_str = ", ".join([f"{source} ({count})" for source, count in top_sources])
            differences.append(f"Coverage spans multiple sources: {sources_str}")
        
        # If no significant differences found
        if not differences:
            differences.append("Coverage is relatively consistent across articles")
        
        return differences

    def _get_final_sentiment(self, distribution: Dict[str, Any], articles: List[Dict[str, Any]]) -> str:
        """Generate final sentiment analysis based on distribution and article content."""
        try:
            # Get basic sentiment counts
            basic_dist = distribution.get('basic', {})
            positive_count = basic_dist.get('positive', 0)
            negative_count = basic_dist.get('negative', 0)
            neutral_count = basic_dist.get('neutral', 0)
            
            total_articles = positive_count + negative_count + neutral_count
            
            if total_articles == 0:
                return "No sentiment data available"
            
            # Calculate percentages
            positive_pct = (positive_count / total_articles) * 100
            negative_pct = (negative_count / total_articles) * 100
            neutral_pct = (neutral_count / total_articles) * 100
            
            # Get average sentiment score
            avg_score = distribution.get('avg_score', 0.5)
            
            # Get volatility
            volatility = distribution.get('volatility', 0)
            
            # Get sentiment indices
            indices = distribution.get('sentiment_indices', {})
            positivity_index = indices.get('positivity_index', 0.5)
            negativity_index = indices.get('negativity_index', 0.5)
            emotional_intensity = indices.get('emotional_intensity', 0)
            controversy_score = indices.get('controversy_score', 0)
            esg_relevance = indices.get('esg_relevance', 0)
            
            # Generate analysis text
            analysis = []
            
            # Overall sentiment
            if positive_pct > 60:
                analysis.append(f"Overall sentiment is predominantly positive ({positive_pct:.1f}%).")
            elif negative_pct > 60:
                analysis.append(f"Overall sentiment is predominantly negative ({negative_pct:.1f}%).")
            elif neutral_pct > 60:
                analysis.append(f"Overall sentiment is predominantly neutral ({neutral_pct:.1f}%).")
            elif positive_pct > negative_pct and positive_pct > neutral_pct:
                analysis.append(f"Overall sentiment leans positive ({positive_pct:.1f}%), with some mixed coverage.")
            elif negative_pct > positive_pct and negative_pct > neutral_pct:
                analysis.append(f"Overall sentiment leans negative ({negative_pct:.1f}%), with some mixed coverage.")
            else:
                analysis.append(f"Sentiment is mixed across sources (Positive: {positive_pct:.1f}%, Negative: {negative_pct:.1f}%, Neutral: {neutral_pct:.1f}%).")
            
            # Sentiment indices insights
            if positivity_index > 0.7:
                analysis.append(f"High positivity index ({positivity_index:.2f}) indicates strong positive sentiment.")
            elif positivity_index < 0.3 and negativity_index > 0.7:
                analysis.append(f"High negativity index ({negativity_index:.2f}) with low positivity suggests strongly negative coverage.")
            
            if emotional_intensity > 0.6:
                analysis.append(f"Coverage shows high emotional intensity ({emotional_intensity:.2f}).")
            
            if controversy_score > 0.5:
                analysis.append(f"Coverage shows significant controversy ({controversy_score:.2f}), with polarized opinions.")
            
            if esg_relevance > 0.4:
                analysis.append(f"Coverage includes significant ESG-related content ({esg_relevance:.2f}).")
            
            # Volatility
            if volatility > 0.2:
                analysis.append(f"Sentiment varies considerably across articles (volatility: {volatility:.2f}).")
            else:
                analysis.append(f"Sentiment is relatively consistent across articles (volatility: {volatility:.2f}).")
            
            return " ".join(analysis)
            
        except Exception as e:
            print(f"Error generating final sentiment: {str(e)}")
            return "Unable to generate final sentiment analysis due to an error."
