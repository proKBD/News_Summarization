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
                continue
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
                continue
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
        self.sentiment_pipeline = pipeline("sentiment-analysis", 
                                      model=SENTIMENT_MODEL)
        self.summarizer = pipeline("summarization", 
                               model=SUMMARIZATION_MODEL)
        self.vectorizer = TfidfVectorizer(stop_words='english', 
                                      max_features=10)
        
    def analyze_article(self, article: Dict[str, str]) -> Dict[str, Any]:
        """Analyze sentiment and generate summary for an article."""
        try:
            # Get the full text by combining title and content
            full_text = f"{article['title']} {article['content']}"
            
            # Generate summary
            summary = self.summarize_text(full_text)
            
            # Analyze sentiment
            sentiment_result = self.sentiment_pipeline(full_text)[0]
            sentiment_label = sentiment_result['label'].lower()
            sentiment_score = round(sentiment_result['score'], 3)
            
            # Extract key topics
            topics = self.extract_topics(full_text)
            
            # Add analysis to article
            analyzed_article = article.copy()
            analyzed_article.update({
                'summary': summary,
                'sentiment': sentiment_label,
                'sentiment_score': sentiment_score,
                'topics': topics,
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
                'topics': [],
                'analysis_timestamp': datetime.now().isoformat()
            })
            return article

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
    def analyze_coverage(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comparative analysis across articles."""
        if not articles:
            return {
                "sentiment_distribution": {},
                "common_topics": [],
                "source_distribution": {},
                "coverage_differences": [],
                "final_sentiment": "No articles found for analysis.",
                "total_articles": 0
            }

        # Calculate sentiment distribution
        sentiment_dist = self._get_sentiment_distribution(articles)
        
        # Get topic analysis
        topic_analysis = self._analyze_topics(articles)
        
        # Get source distribution
        source_dist = {}
        for article in articles:
            source = article.get('source', 'Unknown')
            source_dist[source] = source_dist.get(source, 0) + 1
            
        # Get final sentiment analysis
        final_sentiment = self._get_final_sentiment(sentiment_dist, articles)
        
        # Compare article differences
        coverage_diff = self._analyze_coverage_differences(articles)
        
        return {
            "sentiment_distribution": sentiment_dist,
            "common_topics": topic_analysis,
            "source_distribution": source_dist,
            "coverage_differences": coverage_diff,
            "final_sentiment": final_sentiment,
            "total_articles": len(articles)
        }

    def _get_sentiment_distribution(self, articles: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of sentiments across articles."""
        distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
        for article in articles:
            sentiment = article.get('sentiment', 'neutral').lower()
            distribution[sentiment] = distribution.get(sentiment, 0) + 1
        return distribution

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
            sorted_indices = avg_scores.argsort()[::-1]
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
            differences.append(f"Coverage shows varied sentiments: {', '.join(unique_sentiments)}")
            
        # Compare source differences
        sources = [article.get('source', 'Unknown') for article in articles]
        if len(set(sources)) > 1:
            differences.append(f"Coverage from {len(set(sources))} different news sources")
            
        # Compare publication dates if available
        dates = []
        for article in articles:
            if 'date' in article:
                try:
                    dates.append(article['date'])
                except:
                    continue
        
        if dates:
            date_range = f"{min(dates)} to {max(dates)}"
            differences.append(f"Articles span from {date_range}")
        
        return differences

    def _get_final_sentiment(self, distribution: Dict[str, int], articles: List[Dict[str, Any]]) -> str:
        """Generate final sentiment analysis based on distribution and article content."""
        if not articles:
            return "No articles available for analysis."
            
        total = sum(distribution.values())
        if total == 0:
            return "No sentiment analysis available."
            
        # Calculate sentiment percentages
        sentiment_percentages = {k: (v / total) * 100 for k, v in distribution.items()}
        dominant_sentiment = max(sentiment_percentages.items(), key=lambda x: x[1])[0]
        
        # Get key topics and sources
        topics = set()
        sources = set()
        for article in articles:
            if 'topics' in article:
                topics.update(article['topics'])
            if 'source' in article:
                sources.add(article['source'])
        
        # Generate dynamic summary
        summary_parts = []
        
        # Sentiment distribution summary
        sentiment_desc = f"Based on analysis of {total} articles from {len(sources)} different sources, "
        if sentiment_percentages[dominant_sentiment] > 60:
            sentiment_desc += f"there is a strong {dominant_sentiment} sentiment ({sentiment_percentages[dominant_sentiment]:.1f}% of articles)"
        else:
            sentiment_desc += f"the overall sentiment leans {dominant_sentiment}"
        summary_parts.append(sentiment_desc + ".")
        
        # Market impact and trend analysis
        positive_ratio = sentiment_percentages.get('positive', 0)
        negative_ratio = sentiment_percentages.get('negative', 0)
        neutral_ratio = sentiment_percentages.get('neutral', 0)
        
        if positive_ratio > negative_ratio + 20:
            summary_parts.append(f"Recent coverage indicates strong market confidence, with {positive_ratio:.1f}% positive articles discussing key developments and growth prospects.")
        elif negative_ratio > positive_ratio + 20:
            summary_parts.append(f"Market sentiment shows concerns, with {negative_ratio:.1f}% negative coverage highlighting potential challenges and risks.")
        else:
            summary_parts.append(f"The market shows balanced perspectives with mixed coverage ({positive_ratio:.1f}% positive, {negative_ratio:.1f}% negative, {neutral_ratio:.1f}% neutral).")
        
        return " ".join(summary_parts)
