"""Streamlit frontend for the News Summarization application."""

import streamlit as st
import requests
import pandas as pd
import json
from config import API_BASE_URL
import os
import plotly.express as px

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
                            
                            # Enhanced sentiment display
                            if "sentiment" in article:
                                sentiment_col1, sentiment_col2 = st.columns(2)
                                with sentiment_col1:
                                    st.write("**Sentiment:**", article["sentiment"])
                                    st.write("**Confidence Score:**", f"{article.get('sentiment_score', 0)*100:.1f}%")
                                
                                with sentiment_col2:
                                    # Display fine-grained sentiment if available
                                    if "fine_grained_sentiment" in article and article["fine_grained_sentiment"]:
                                        fine_grained = article["fine_grained_sentiment"]
                                        if "category" in fine_grained:
                                            st.write("**Detailed Sentiment:**", fine_grained["category"])
                                        if "confidence" in fine_grained:
                                            st.write("**Confidence:**", f"{fine_grained['confidence']*100:.1f}%")
                            
                            # Display entities if available
                            if "entities" in article and article["entities"]:
                                st.markdown("**Named Entities:**")
                                entities = article["entities"]
                                
                                # Organizations
                                if "ORG" in entities and entities["ORG"]:
                                    st.write("**Organizations:**", ", ".join(entities["ORG"]))
                                
                                # People
                                if "PERSON" in entities and entities["PERSON"]:
                                    st.write("**People:**", ", ".join(entities["PERSON"]))
                                
                                # Locations
                                if "GPE" in entities and entities["GPE"]:
                                    st.write("**Locations:**", ", ".join(entities["GPE"]))
                                
                                # Money
                                if "MONEY" in entities and entities["MONEY"]:
                                    st.write("**Financial Values:**", ", ".join(entities["MONEY"]))
                            
                            # Display sentiment targets if available
                            if "sentiment_targets" in article and article["sentiment_targets"]:
                                st.markdown("**Sentiment Targets:**")
                                targets = article["sentiment_targets"]
                                for target in targets:
                                    st.markdown(f"**{target['entity']}** ({target['type']}): {target['sentiment']} ({target['confidence']*100:.1f}%)")
                                    st.markdown(f"> {target['context']}")
                                    st.markdown("---")
                            
                            if "url" in article:
                                st.write("**[Read More](%s)**" % article["url"])
                    
                    # Display Comparative Analysis
                    st.header("📊 Comparative Analysis")
                    analysis = result.get("comparative_sentiment_score", {})
                    
                    # Sentiment Distribution
                    if "sentiment_distribution" in analysis:
                        st.subheader("Sentiment Distribution")
                        
                        # Check if sentiment_distribution is a dictionary with a 'basic' key (new format)
                        if isinstance(analysis["sentiment_distribution"], dict) and "basic" in analysis["sentiment_distribution"]:
                            # New format with basic and fine-grained sentiment
                            basic_dist = analysis["sentiment_distribution"]["basic"]
                            dist_df = pd.DataFrame.from_dict(
                                basic_dist, 
                                orient='index',
                                columns=['Count']
                            )
                            
                            # Create a custom pie chart with colors
                            fig = px.pie(
                                dist_df, 
                                values='Count',
                                names=dist_df.index,
                                color=dist_df.index,
                                color_discrete_map={
                                    'positive': 'green',
                                    'negative': 'red',
                                    'neutral': 'yellow'
                                },
                                title="Basic Sentiment Distribution"
                            )
                            # Add percentage labels
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig)
                            
                            # Display fine-grained sentiment if available
                            if "fine_grained" in analysis["sentiment_distribution"] and analysis["sentiment_distribution"]["fine_grained"]:
                                st.subheader("Fine-Grained Sentiment")
                                fine_grained = analysis["sentiment_distribution"]["fine_grained"]
                                fine_df = pd.DataFrame.from_dict(
                                    fine_grained,
                                    orient='index',
                                    columns=['Count']
                                )
                                
                                # Create color mapping for fine-grained sentiment
                                color_map = {}
                                for category in fine_df.index:
                                    if 'positive' in category.lower():
                                        color_map[category] = 'green'
                                    elif 'negative' in category.lower():
                                        color_map[category] = 'red'
                                    else:
                                        color_map[category] = 'yellow'
                                
                                # Create a custom bar chart with colors
                                fig_fine = px.bar(
                                    fine_df, 
                                    x=fine_df.index, 
                                    y='Count',
                                    color=fine_df.index,
                                    color_discrete_map=color_map
                                )
                                st.plotly_chart(fig_fine)
                                
                            # Display sentiment metrics
                            metrics_col1, metrics_col2 = st.columns(2)
                            with metrics_col1:
                                if "avg_score" in analysis["sentiment_distribution"]:
                                    st.metric("Average Sentiment Score", f"{analysis['sentiment_distribution']['avg_score']:.2f}")
                            with metrics_col2:
                                if "volatility" in analysis["sentiment_distribution"]:
                                    st.metric("Sentiment Volatility", f"{analysis['sentiment_distribution']['volatility']:.2f}")
                        else:
                            # Old format (simple dictionary)
                            dist_df = pd.DataFrame.from_dict(
                                analysis["sentiment_distribution"], 
                                orient='index',
                                columns=['Count']
                            )
                            
                            # Create color mapping
                            color_map = {}
                            for category in dist_df.index:
                                if 'positive' in category.lower():
                                    color_map[category] = 'green'
                                elif 'negative' in category.lower():
                                    color_map[category] = 'red'
                                else:
                                    color_map[category] = 'yellow'
                            
                            # Create a custom pie chart with colors
                            fig = px.pie(
                                dist_df, 
                                values='Count',
                                names=dist_df.index,
                                color=dist_df.index,
                                color_discrete_map=color_map,
                                title="Sentiment Distribution"
                            )
                            # Add percentage labels
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig)
                    
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
                        
                        # Display ensemble model information if available
                        if "ensemble_info" in result:
                            with st.expander("Ensemble Model Details"):
                                ensemble = result["ensemble_info"]
                                
                                # Model agreement
                                if "agreement" in ensemble:
                                    st.metric("Model Agreement", f"{ensemble['agreement']*100:.1f}%")
                                
                                # Individual model results
                                if "models" in ensemble:
                                    st.subheader("Individual Model Results")
                                    models_data = []
                                    for model_name, model_info in ensemble["models"].items():
                                        models_data.append({
                                            "Model": model_name,
                                            "Sentiment": model_info.get("sentiment", "N/A"),
                                            "Confidence": f"{model_info.get('confidence', 0)*100:.1f}%"
                                        })
                                    
                                    if models_data:
                                        st.table(pd.DataFrame(models_data))
                        
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
