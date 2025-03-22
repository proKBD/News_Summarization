"""Streamlit frontend for the News Summarization application."""

import streamlit as st
import requests
import pandas as pd
import json
from config import API_BASE_URL
import os
import plotly.express as px
import altair as alt

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
            # Print the response data for debugging
            print("API Response Data:")
            print(json.dumps(data, indent=2))
            
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
                            
                            # Display sentiment indices if available
                            if "sentiment_indices" in article and article["sentiment_indices"]:
                                st.markdown("**Sentiment Indices:**")
                                indices = article["sentiment_indices"]
                                
                                # Create columns for displaying indices
                                idx_cols = st.columns(3)
                                
                                # Display positivity and negativity in first column
                                with idx_cols[0]:
                                    if "positivity_index" in indices:
                                        st.markdown(f"**Positivity:** {indices['positivity_index']:.2f}")
                                    if "negativity_index" in indices:
                                        st.markdown(f"**Negativity:** {indices['negativity_index']:.2f}")
                                
                                # Display emotional intensity and controversy in second column
                                with idx_cols[1]:
                                    if "emotional_intensity" in indices:
                                        st.markdown(f"**Emotional Intensity:** {indices['emotional_intensity']:.2f}")
                                    if "controversy_score" in indices:
                                        st.markdown(f"**Controversy:** {indices['controversy_score']:.2f}")
                                
                                # Display confidence and ESG in third column
                                with idx_cols[2]:
                                    if "confidence_score" in indices:
                                        st.markdown(f"**Confidence:** {indices['confidence_score']:.2f}")
                                    if "esg_relevance" in indices:
                                        st.markdown(f"**ESG Relevance:** {indices['esg_relevance']:.2f}")
                            
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
                        
                        # Debug: Print sentiment distribution data
                        print("Sentiment Distribution Data:")
                        print(json.dumps(analysis["sentiment_distribution"], indent=2))
                        
                        sentiment_dist = analysis["sentiment_distribution"]
                        
                        # Create a very simple visualization that will definitely work
                        try:
                            # Extract basic sentiment data
                            if isinstance(sentiment_dist, dict):
                                if "basic" in sentiment_dist and isinstance(sentiment_dist["basic"], dict):
                                    basic_dist = sentiment_dist["basic"]
                                elif any(k in sentiment_dist for k in ['positive', 'negative', 'neutral']):
                                    basic_dist = {k: v for k, v in sentiment_dist.items() 
                                                if k in ['positive', 'negative', 'neutral']}
                                else:
                                    basic_dist = {'positive': 0, 'negative': 0, 'neutral': 1}
                            else:
                                basic_dist = {'positive': 0, 'negative': 0, 'neutral': 1}
                            
                            # Calculate percentages
                            total_articles = sum(basic_dist.values())
                            if total_articles > 0:
                                percentages = {
                                    k: (v / total_articles) * 100 
                                    for k, v in basic_dist.items()
                                }
                            else:
                                percentages = {k: 0 for k in basic_dist}
                            
                            # Display as simple text and metrics
                            st.write("**Sentiment Distribution:**")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Positive", 
                                    basic_dist.get('positive', 0),
                                    f"{percentages.get('positive', 0):.1f}%"
                                )
                            with col2:
                                st.metric(
                                    "Negative", 
                                    basic_dist.get('negative', 0),
                                    f"{percentages.get('negative', 0):.1f}%"
                                )
                            with col3:
                                st.metric(
                                    "Neutral", 
                                    basic_dist.get('neutral', 0),
                                    f"{percentages.get('neutral', 0):.1f}%"
                                )
                            
                            # Create a simple bar chart using Altair
                            
                            # Create a simple DataFrame with consistent capitalization and percentages
                            chart_data = pd.DataFrame({
                                'Sentiment': ['Positive', 'Negative', 'Neutral'],
                                'Count': [
                                    basic_dist.get('positive', 0),  # Map lowercase keys to capitalized display
                                    basic_dist.get('negative', 0),
                                    basic_dist.get('neutral', 0)
                                ],
                                'Percentage': [
                                    f"{percentages.get('positive', 0):.1f}%",
                                    f"{percentages.get('negative', 0):.1f}%",
                                    f"{percentages.get('neutral', 0):.1f}%"
                                ]
                            })
                            
                            # Add debug output to see what's in the data
                            print("Chart Data for Sentiment Distribution:")
                            print(chart_data)
                            
                            # Create a simple bar chart with percentages
                            chart = alt.Chart(chart_data).mark_bar().encode(
                                y='Sentiment',  # Changed from x to y for horizontal bars
                                x='Count',      # Changed from y to x for horizontal bars
                                color=alt.Color('Sentiment', scale=alt.Scale(
                                    domain=['Positive', 'Negative', 'Neutral'],
                                    range=['green', 'red', 'gray']
                                )),
                                tooltip=['Sentiment', 'Count', 'Percentage']  # Add tooltip with percentage
                            ).properties(
                                width=600,
                                height=300
                            )
                            
                            # Add text labels with percentages
                            text = chart.mark_text(
                                align='left',
                                baseline='middle',
                                dx=3  # Nudge text to the right so it doesn't overlap with the bar
                            ).encode(
                                text='Percentage'
                            )
                            
                            # Combine the chart and text
                            chart_with_text = (chart + text)
                            
                            st.altair_chart(chart_with_text, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"Error creating visualization: {str(e)}")
                            st.write("Fallback to simple text display:")
                            if isinstance(sentiment_dist, dict):
                                if "basic" in sentiment_dist:
                                    st.write(f"Positive: {sentiment_dist['basic'].get('positive', 0)}")
                                    st.write(f"Negative: {sentiment_dist['basic'].get('negative', 0)}")
                                    st.write(f"Neutral: {sentiment_dist['basic'].get('neutral', 0)}")
                                else:
                                    st.write(f"Positive: {sentiment_dist.get('positive', 0)}")
                                    st.write(f"Negative: {sentiment_dist.get('negative', 0)}")
                                    st.write(f"Neutral: {sentiment_dist.get('neutral', 0)}")
                            else:
                                st.write("No valid sentiment data available")
                    
                    # Display sentiment indices if available
                    if "sentiment_indices" in analysis and analysis["sentiment_indices"]:
                        st.subheader("Sentiment Indices")
                        
                        # Debug: Print sentiment indices
                        print("Sentiment Indices:")
                        print(json.dumps(analysis["sentiment_indices"], indent=2))
                        
                        # Get the indices data
                        indices = analysis["sentiment_indices"]
                        
                        # Create a very simple visualization that will definitely work
                        try:
                            if isinstance(indices, dict):
                                # Display as simple metrics in columns
                                cols = st.columns(3)
                                
                                # Define display names and descriptions
                                display_names = {
                                    "positivity_index": "Positivity",
                                    "negativity_index": "Negativity",
                                    "emotional_intensity": "Emotional Intensity",
                                    "controversy_score": "Controversy",
                                    "confidence_score": "Confidence",
                                    "esg_relevance": "ESG Relevance"
                                }
                                
                                # Display each index as a metric
                                for i, (key, value) in enumerate(indices.items()):
                                    if isinstance(value, (int, float)):
                                        with cols[i % 3]:
                                            display_name = display_names.get(key, key.replace("_", " ").title())
                                            st.metric(display_name, f"{value:.2f}")
                                
                                # Create a simple bar chart using Altair
                                
                                # Create a simple DataFrame
                                chart_data = pd.DataFrame({
                                    'Index': [display_names.get(k, k.replace("_", " ").title()) for k in indices.keys()],
                                    'Value': [v if isinstance(v, (int, float)) else 0 for v in indices.values()]
                                })
                                
                                # Create a simple bar chart
                                chart = alt.Chart(chart_data).mark_bar().encode(
                                    x='Value',
                                    y='Index',
                                    color=alt.Color('Index')
                                ).properties(
                                    width=600,
                                    height=300
                                )
                                
                                st.altair_chart(chart, use_container_width=True)
                                
                                # Add descriptions
                                with st.expander("Sentiment Indices Explained"):
                                    st.markdown("""
                                    - **Positivity**: Measures the positive sentiment in the articles (0-1)
                                    - **Negativity**: Measures the negative sentiment in the articles (0-1)
                                    - **Emotional Intensity**: Measures the overall emotional content (0-1)
                                    - **Controversy**: High when both positive and negative sentiments are strong (0-1)
                                    - **Confidence**: Confidence in the sentiment analysis (0-1)
                                    - **ESG Relevance**: Relevance to Environmental, Social, and Governance topics (0-1)
                                    """)
                            else:
                                st.warning("Sentiment indices data is not in the expected format.")
                                st.write("No valid sentiment indices available")
                        except Exception as e:
                            st.error(f"Error creating indices visualization: {str(e)}")
                            st.write("Fallback to simple text display:")
                            if isinstance(indices, dict):
                                for key, value in indices.items():
                                    if isinstance(value, (int, float)):
                                        st.write(f"{key.replace('_', ' ').title()}: {value:.2f}")
                            else:
                                st.write("No valid sentiment indices data available")
                    
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
                        
                        # Display sentiment indices in the sidebar if available
                        if "sentiment_indices" in analysis and analysis["sentiment_indices"]:
                            indices = analysis["sentiment_indices"]
                            # Verify we have valid data
                            if indices and any(isinstance(v, (int, float)) for v in indices.values()):
                                st.sidebar.markdown("### Sentiment Indices")
                                for idx_name, idx_value in indices.items():
                                    if isinstance(idx_value, (int, float)):
                                        formatted_name = " ".join(word.capitalize() for word in idx_name.replace("_", " ").split())
                                        st.sidebar.metric(formatted_name, f"{idx_value:.2f}")
                        
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
