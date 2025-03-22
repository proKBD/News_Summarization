# News Summarization and Text-to-Speech Application

This application extracts news articles about a company, performs sentiment analysis, and generates a Hindi text-to-speech summary.

## Features

- News article extraction using BeautifulSoup
- Sentiment analysis of articles
- Comparative sentiment analysis across articles
- Text-to-Speech conversion to Hindi
- Web interface using Streamlit
- RESTful API backend

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd News_summarization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
# Start the API server
uvicorn api:app --reload --port 8000

# In a new terminal, start the Streamlit interface
streamlit run app.py
```

## API Documentation

The application exposes the following API endpoints:

- `POST /api/analyze`: Analyze news articles for a company
- `GET /api/companies`: Get list of supported companies
- `POST /api/generate-audio`: Generate Hindi TTS for a summary

## Models Used

- Sentiment Analysis: DistilBERT fine-tuned on financial news
- Text Summarization: T5-base fine-tuned for news summarization
- TTS: gTTS (Google Text-to-Speech)

## System Limitations

- Only supports non-JavaScript enabled news websites
- Limited to English language news articles for analysis
- Audio generation may take some time for longer summaries

## Project Structure

```
News_summarization/
├── app.py              # Streamlit frontend
├── api.py             # FastAPI backend
├── utils.py           # Utility functions
├── config.py          # Configuration settings
├── requirements.txt   # Project dependencies
└── README.md          # Documentation
```

## License

MIT License

## Documentation

### Project Setup

#### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Internet connection for accessing news sources and models

#### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/proKBD/News_Summarization.git
   cd News_Summarization
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download required models (if not automatically downloaded):

   ```bash
   python -m spacy download en_core_web_sm
   ```

#### Running the Application

1. Start the API server:

   ```bash
   python api.py
   ```

   The API server will run on [http://localhost:8005](http://localhost:8005)

2. Start the Streamlit frontend:

   ```bash
   streamlit run app.py
   ```

   The Streamlit app will be available at [http://localhost:8501](http://localhost:8501)

### Model Details

#### Summarization Models

- **Primary Model**: `facebook/bart-large-cnn`
  - Architecture: BART (Bidirectional and Auto-Regressive Transformers)
  - Purpose: Generates concise summaries of news articles
  - Features: Maintains key information while reducing text length by 70-80%

#### Sentiment Analysis Models

1. **Primary Sentiment Model**: `finiteautomata/bertweet-base-sentiment-analysis`
   - Architecture: BERTweet (BERT model trained on Twitter data)
   - Purpose: Classifies text as positive, negative, or neutral
   - Features: Optimized for short-form content similar to news headlines

2. **Fine-Grained Sentiment Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
   - Architecture: RoBERTa (Robustly optimized BERT approach)
   - Purpose: Provides more detailed sentiment categories
   - Features: Detects subtle emotional nuances in text

3. **Ensemble Approach**:
   - The application combines multiple sentiment models:
     - Transformer-based models (BERTweet, RoBERTa)
     - TextBlob (lexicon-based approach)
     - VADER (rule-based sentiment analyzer)
   - This ensemble approach improves accuracy and robustness

#### TTS Model Integration

- **gTTS (Google Text-to-Speech)**
  - Purpose: Converts the final analysis into spoken audio
  - Features: Multi-language support, natural-sounding speech

### API Development

#### API Architecture

The application uses FastAPI to create a RESTful API that serves as the backend for the Streamlit frontend. The API handles:

- News extraction from multiple sources
- Sentiment analysis of articles
- Comparative analysis across sources
- Text-to-speech conversion

#### API Endpoints

1. **Health Check**
   - **Endpoint**: `/health`
   - **Method**: GET
   - **Purpose**: Verify API is running

2. **Search News**
   - **Endpoint**: `/search_news`
   - **Method**: POST
   - **Purpose**: Find news articles about a company
   - **Parameters**: `company_name` (string)

3. **Analyze Article**
   - **Endpoint**: `/analyze_article`
   - **Method**: POST
   - **Purpose**: Perform sentiment analysis on a single article
   - **Parameters**: `article` (object with title, url, source, date, content)

4. **Analyze Company**
   - **Endpoint**: `/analyze_company`
   - **Method**: POST
   - **Purpose**: Comprehensive analysis of news coverage
   - **Parameters**: `company_name` (string)

5. **Generate Audio**
   - **Endpoint**: `/generate_audio`
   - **Method**: POST
   - **Purpose**: Convert text to speech
   - **Parameters**: `text` (string), `filename` (string)

#### Accessing the API

##### Using Postman

1. Download and install [Postman](https://www.postman.com/downloads/)
2. Create a new request:
   - Set the HTTP method (GET/POST)
   - Enter the endpoint URL (e.g., `http://localhost:8005/analyze_company`)
   - For POST requests:
     - Go to the "Body" tab
     - Select "raw" and "JSON"
     - Enter the required JSON payload (e.g., `{"company_name": "Microsoft"}`)
3. Click "Send" to execute the request

##### Using curl

Example for the analyze_company endpoint:

```bash
curl -X POST http://localhost:8005/analyze_company \
  -H "Content-Type: application/json" \
  -d '{"company_name": "Apple"}'
```

##### Using Python requests

```python
import requests
import json

url = "http://localhost:8005/analyze_company"
payload = {"company_name": "Tesla"}
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(payload), headers=headers)
result = response.json()
print(result)
```

### API Usage (Third-Party)

#### News Source APIs

- **Purpose**: Extract news articles about specified companies
- **Integration**: Custom web scraping with appropriate headers
- **Sources**:
  - Google News
  - Bing News
  - Yahoo News
  - Reuters
  - MarketWatch
  - Investing.com
  - TechCrunch
  - ZDNet

#### Hugging Face API

- **Purpose**: Access pre-trained NLP models
- **Integration**: Using the transformers library
- **Models Used**:
  - Sentiment analysis models
  - Summarization models
  - Named entity recognition

#### External TTS API Integration

- **Purpose**: Convert text analysis to audio
- **Integration**: Using the gTTS library
- **Features**: Multi-language support, natural voice

### Assumptions & Limitations

#### Assumptions

1. **Internet Connectivity**: The application assumes constant internet connectivity to access news sources and models.
2. **English Content**: The application is optimized for English language news articles.
3. **News Availability**: The application assumes that relevant news articles exist for the queried company.
4. **Model Availability**: The application assumes that the required Hugging Face models are accessible.
5. **Processing Power**: The application assumes sufficient CPU/GPU resources for running transformer models.

#### Limitations

1. **News Source Restrictions**: Some news sources may block requests if too many are made in a short period.
2. **Model Size**: The transformer models require significant memory (4-8GB RAM recommended).
3. **Processing Time**: Comprehensive analysis may take 30-60 seconds depending on the number of articles.
4. **Accuracy Boundaries**: Sentiment analysis has inherent limitations in detecting sarcasm, irony, or context-specific sentiment.
5. **API Rate Limits**: Third-party APIs may have rate limits that restrict usage.
6. **Content Depth**: The analysis is based on available news articles, which may not cover all aspects of a company.

#### Future Improvements

1. Implement caching to reduce repeated API calls
2. Add support for additional languages
3. Incorporate more news sources for broader coverage
4. Implement user authentication for API access
5. Add more fine-grained sentiment analysis categories
6. Improve error handling for unreliable news sources
