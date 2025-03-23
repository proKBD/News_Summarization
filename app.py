"""Streamlit frontend for the News Summarization application."""

import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import altair as alt
from utils import (
    analyze_company_data,
    TextToSpeechConverter,
    get_translator,
    NewsExtractor,
    SentimentAnalyzer,
    TextSummarizer
)

# Set page config
st.set_page_config(
    page_title="News Summarization App",
    page_icon="📰",
    layout="wide"
)

# Show loading message
with st.spinner("Initializing the application... Please wait while we load the models."):
    # Initialize components
    try:
        st.success("Application initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing application: {str(e)}")
        st.info("Please try refreshing the page.")

def process_company(company_name):
    """Process company data directly."""
    try:
        # Call the analysis function directly from utils
        data = analyze_company_data(company_name)
        
        # Generate Hindi audio from final analysis
        if data.get("final_sentiment_analysis"):
            # Get the translator
            translator = get_translator()
            if translator:
                try:
                    # Create a more detailed Hindi explanation
                    sentiment_explanation = f"""
                    {company_name} के समाचारों का विश्लेषण:
                    
                    समग्र भावना: {data['final_sentiment_analysis']}
                    
                    भावनात्मक विश्लेषण:
                    - सकारात्मक भावना: {data.get('comparative_sentiment_score', {}).get('sentiment_indices', {}).get('positivity_index', 0):.2f}
                    - नकारात्मक भावना: {data.get('comparative_sentiment_score', {}).get('sentiment_indices', {}).get('negativity_index', 0):.2f}
                    - भावनात्मक तीव्रता: {data.get('comparative_sentiment_score', {}).get('sentiment_indices', {}).get('emotional_intensity', 0):.2f}
                    
                    विश्वसनीयता स्कोर: {data.get('comparative_sentiment_score', {}).get('sentiment_indices', {}).get('confidence_score', 0):.2f}
                    """
                    
                    # Generate Hindi audio
                    tts_converter = TextToSpeechConverter()
                    audio_path = tts_converter.generate_audio(
                        sentiment_explanation,
                        f'{company_name}_summary'
                    )
                    data['audio_path'] = audio_path
                except Exception as e:
                    print(f"Error generating Hindi audio: {str(e)}")
                    data['audio_path'] = None
            else:
                print("Translator not available")
                data['audio_path'] = None
            
        return data
    except Exception as e:
        st.error(f"Error processing company: {str(e)}")
        return {"articles": [], "comparative_sentiment_score": {}, "final_sentiment_analysis": "", "audio_path": None}

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
                try:
                    # Process company data
                    data = process_company(company)
                    
                    if not data["articles"]:
                        st.error("No articles found for analysis.")
                        return
                    
                    # Display Articles
                    st.header("📑 News Articles")
                    for idx, article in enumerate(data["articles"], 1):
                        with st.expander(f"Article {idx}: {article['title']}"):
                            # Display content with proper formatting
                            if article.get("content"):
                                st.markdown("**Content:**")
                                st.write(article["content"])
                            else:
                                st.warning("No content available for this article")
                            
                            # Display summary if available
                            if article.get("summary"):
                                st.markdown("**Summary:**")
                                st.write(article["summary"])
                            
                            # Display source
                            if article.get("source"):
                                st.markdown("**Source:**")
                                st.write(article["source"])
                            
                            # Enhanced sentiment display
                            if "sentiment" in article:
                                sentiment_col1, sentiment_col2 = st.columns(2)
                                with sentiment_col1:
                                    st.markdown("**Basic Sentiment:**")
                                    st.write(article["sentiment"])
                                    if "sentiment_score" in article:
                                        st.write(f"**Confidence Score:** {article['sentiment_score']*100:.1f}%")
                                
                                with sentiment_col2:
                                    # Display fine-grained sentiment if available
                                    if "fine_grained_sentiment" in article and article["fine_grained_sentiment"]:
                                        st.markdown("**Detailed Sentiment:**")
                                        fine_grained = article["fine_grained_sentiment"]
                                        if "category" in fine_grained:
                                            st.write(f"Category: {fine_grained['category']}")
                                        if "confidence" in fine_grained:
                                            st.write(f"Confidence: {fine_grained['confidence']*100:.1f}%")
                            
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
                            
                            # Display URL if available
                            if "url" in article:
                                st.markdown(f"**[Read More]({article['url']})**")
                    
                    # Display Comparative Analysis
                    st.header("📊 Comparative Analysis")
                    analysis = data.get("comparative_sentiment_score", {})
                    
                    # Sentiment Distribution
                    if "sentiment_distribution" in analysis:
                        st.subheader("Sentiment Distribution")
                        
                        sentiment_dist = analysis["sentiment_distribution"]
                        
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
                            
                            # Display as metrics
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
                            
                            # Create visualization
                            chart_data = pd.DataFrame({
                                'Sentiment': ['Positive', 'Negative', 'Neutral'],
                                'Count': [
                                    basic_dist.get('positive', 0),
                                    basic_dist.get('negative', 0),
                                    basic_dist.get('neutral', 0)
                                ],
                                'Percentage': [
                                    f"{percentages.get('positive', 0):.1f}%",
                                    f"{percentages.get('negative', 0):.1f}%",
                                    f"{percentages.get('neutral', 0):.1f}%"
                                ]
                            })
                            
                            chart = alt.Chart(chart_data).mark_bar().encode(
                                y='Sentiment',
                                x='Count',
                                color=alt.Color('Sentiment', scale=alt.Scale(
                                    domain=['Positive', 'Negative', 'Neutral'],
                                    range=['green', 'red', 'gray']
                                )),
                                tooltip=['Sentiment', 'Count', 'Percentage']
                            ).properties(
                                width=600,
                                height=300
                            )
                            
                            text = chart.mark_text(
                                align='left',
                                baseline='middle',
                                dx=3
                            ).encode(
                                text='Percentage'
                            )
                            
                            chart_with_text = (chart + text)
                            st.altair_chart(chart_with_text, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"Error creating visualization: {str(e)}")
                    
                    # Display sentiment indices if available
                    if "sentiment_indices" in analysis and analysis["sentiment_indices"]:
                        st.subheader("Sentiment Indices")
                        
                        indices = analysis["sentiment_indices"]
                        
                        try:
                            if isinstance(indices, dict):
                                # Display as metrics in columns
                                cols = st.columns(3)
                                
                                display_names = {
                                    "positivity_index": "Positivity",
                                    "negativity_index": "Negativity",
                                    "emotional_intensity": "Emotional Intensity",
                                    "controversy_score": "Controversy",
                                    "confidence_score": "Confidence",
                                    "esg_relevance": "ESG Relevance"
                                }
                                
                                for i, (key, value) in enumerate(indices.items()):
                                    if isinstance(value, (int, float)):
                                        with cols[i % 3]:
                                            display_name = display_names.get(key, key.replace("_", " ").title())
                                            st.metric(display_name, f"{value:.2f}")
                                
                                # Create visualization
                                chart_data = pd.DataFrame({
                                    'Index': [display_names.get(k, k.replace("_", " ").title()) for k in indices.keys()],
                                    'Value': [v if isinstance(v, (int, float)) else 0 for v in indices.values()]
                                })
                                
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
                        except Exception as e:
                            st.error(f"Error creating indices visualization: {str(e)}")
                    
                    # Display Final Analysis
                    st.header("📊 Final Analysis")
                    
                    # Display overall sentiment analysis with enhanced formatting
                    if data.get("final_sentiment_analysis"):
                        st.markdown("### Overall Sentiment Analysis")
                        analysis_parts = data["final_sentiment_analysis"].split(". ")
                        if len(analysis_parts) >= 2:
                            # First sentence - Overall sentiment
                            st.markdown(f"**{analysis_parts[0]}.**")
                            # Second sentence - Key findings
                            st.markdown(f"**{analysis_parts[1]}.**")
                            # Third sentence - Additional insights (if available)
                            if len(analysis_parts) > 2:
                                st.markdown(f"**{analysis_parts[2]}.**")
                        else:
                            st.write(data["final_sentiment_analysis"])
                        
                        # Add sentiment strength indicator
                        if data.get("ensemble_info"):
                            ensemble_info = data["ensemble_info"]
                            if "model_agreement" in ensemble_info:
                                agreement = ensemble_info["model_agreement"]
                                strength = "Strong" if agreement > 0.8 else "Moderate" if agreement > 0.6 else "Weak"
                                st.markdown(f"**Sentiment Strength:** {strength} (Agreement: {agreement:.2f})")
                    
                    # Display ensemble model details
                    if data.get("ensemble_info"):
                        st.subheader("Ensemble Model Details")
                        ensemble_info = data["ensemble_info"]
                        
                        # Create columns for model details
                        model_cols = st.columns(3)
                        
                        with model_cols[0]:
                            st.markdown("**Primary Model:**")
                            if "models" in ensemble_info and "transformer" in ensemble_info["models"]:
                                model = ensemble_info["models"]["transformer"]
                                st.write(f"Sentiment: {model['sentiment']}")
                                st.write(f"Score: {model['score']:.3f}")
                        
                        with model_cols[1]:
                            st.markdown("**TextBlob Analysis:**")
                            if "models" in ensemble_info and "textblob" in ensemble_info["models"]:
                                model = ensemble_info["models"]["textblob"]
                                st.write(f"Sentiment: {model['sentiment']}")
                                st.write(f"Score: {model['score']:.3f}")
                        
                        with model_cols[2]:
                            st.markdown("**VADER Analysis:**")
                            if "models" in ensemble_info and "vader" in ensemble_info["models"]:
                                model = ensemble_info["models"]["vader"]
                                st.write(f"Sentiment: {model['sentiment']}")
                                st.write(f"Score: {model['score']:.3f}")
                        
                        # Display ensemble agreement if available
                        if "model_agreement" in ensemble_info:
                            st.markdown(f"**Model Agreement:** {ensemble_info['model_agreement']:.3f}")
                    
                    # Display Hindi audio player
                    st.subheader("🔊 Listen to Analysis (Hindi)")
                    if data.get("audio_path") and os.path.exists(data["audio_path"]):
                        st.audio(data["audio_path"])
                    else:
                        st.info("Generating Hindi audio summary...")
                        with st.spinner("Please wait while we generate the Hindi audio summary..."):
                            # Try to generate audio again
                            translator = get_translator()
                            if translator and data.get("final_sentiment_analysis"):
                                try:
                                    # Translate final analysis to Hindi
                                    translated_analysis = translator.translate(
                                        data["final_sentiment_analysis"],
                                        dest='hi'
                                    ).text
                                    
                                    # Generate Hindi audio
                                    tts_converter = TextToSpeechConverter()
                                    audio_path = tts_converter.generate_audio(
                                        translated_analysis,
                                        f'{company}_summary'
                                    )
                                    if audio_path and os.path.exists(audio_path):
                                        st.audio(audio_path)
                                    else:
                                        st.error("Hindi audio summary not available")
                                except Exception as e:
                                    st.error(f"Error generating Hindi audio: {str(e)}")
                            else:
                                st.error("Hindi audio summary not available")
                    
                    # Total Articles
                    if "total_articles" in analysis:
                        st.sidebar.info(f"Found {analysis['total_articles']} articles")
                
                except Exception as e:
                    st.error(f"Error analyzing company data: {str(e)}")
                    print(f"Error: {str(e)}")

    # Add a disclaimer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.write("This app analyzes news articles and provides sentiment analysis for any company.")

if __name__ == "__main__":
    main()
