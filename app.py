"""Streamlit frontend for the News Summarization application."""

import streamlit as st
import requests
import pandas as pd
import json
from config import API_BASE_URL
import os

st.set_page_config(
    page_title="News Summarization App",
    page_icon="📰",
    layout="wide"
)

def analyze_company(company_name):
    """Send analysis request to API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/analyze",
            json={"name": company_name}
        )
        if response.status_code == 200:
            data = response.json()
            # Download audio file if available
            if 'audio_url' in data:
                audio_response = requests.get(f"{API_BASE_URL}{data['audio_url']}")
                if audio_response.status_code == 200:
                    data['audio_content'] = audio_response.content
            return data
        else:
            st.error(f"Error from API: {response.text}")
            return {"articles": [], "comparative_sentiment_score": {}, "final_sentiment_analysis": "", "audio_url": None}
    except Exception as e:
        st.error(f"Error analyzing company: {str(e)}")
        return {"articles": [], "comparative_sentiment_score": {}, "final_sentiment_analysis": "", "audio_url": None}

def main():
    st.title("📰 News Summarization and Analysis")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Replace dropdown with text input
    company = st.sidebar.text_input(
        "Enter Company Name",
        placeholder="e.g., Tesla, Apple, Microsoft, or any other company",
        help="Enter the name of any company you want to analyze"
    )
    
    if st.sidebar.button("Analyze") and company:
        if len(company.strip()) < 2:
            st.sidebar.error("Please enter a valid company name (at least 2 characters)")
        else:
            with st.spinner("Analyzing news articles..."):
                result = analyze_company(company)
                
                if result and result.get("articles"):
                    # Display Articles
                    st.header("📑 News Articles")
                    for idx, article in enumerate(result["articles"], 1):
                        with st.expander(f"Article {idx}: {article['title']}"):
                            st.write("**Content:**", article.get("content", "No content available"))
                            if "summary" in article:
                                st.write("**Summary:**", article["summary"])
                            st.write("**Source:**", article.get("source", "Unknown"))
                            if "sentiment" in article:
                                st.write("**Sentiment:**", article["sentiment"])
                                st.write("**Confidence Score:**", f"{article.get('sentiment_score', 'N/A')*100:.1f}%")
                            if "url" in article:
                                st.write("**[Read More](%s)**" % article["url"])
                    
                    # Display Comparative Analysis
                    st.header("📊 Comparative Analysis")
                    analysis = result.get("comparative_sentiment_score", {})
                    
                    # Sentiment Distribution
                    if "sentiment_distribution" in analysis:
                        st.subheader("Sentiment Distribution")
                        dist_df = pd.DataFrame.from_dict(
                            analysis["sentiment_distribution"], 
                            orient='index',
                            columns=['Count']
                        )
                        st.bar_chart(dist_df)
                    
                    # Source Distribution
                    if "source_distribution" in analysis:
                        st.subheader("Source Distribution")
                        source_df = pd.DataFrame.from_dict(
                            analysis["source_distribution"],
                            orient='index',
                            columns=['Count']
                        )
                        st.bar_chart(source_df)
                    
                    # Common Topics
                    if "common_topics" in analysis:
                        st.subheader("Common Topics")
                        st.write(", ".join(analysis["common_topics"]) if analysis["common_topics"] else "No common topics found")
                    
                    # Coverage Differences
                    if "coverage_differences" in analysis:
                        st.subheader("Coverage Analysis")
                        for diff in analysis["coverage_differences"]:
                            st.write("- " + diff)
                            
                    # Display Final Sentiment and Audio
                    st.header("🎯 Final Analysis")
                    if "final_sentiment_analysis" in result:
                        st.write(result["final_sentiment_analysis"])
                        
                        # Audio Playback Section
                        st.subheader("🔊 Listen to Analysis (Hindi)")
                        if 'audio_content' in result:
                            st.audio(result['audio_content'], format='audio/mp3')
                        else:
                            st.warning("Hindi audio summary not available")
                    
                    # Total Articles
                    if "total_articles" in analysis:
                        st.sidebar.info(f"Found {analysis['total_articles']} articles")

    # Add a disclaimer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.write("This app analyzes news articles and provides sentiment analysis for any company.")

if __name__ == "__main__":
    main()
