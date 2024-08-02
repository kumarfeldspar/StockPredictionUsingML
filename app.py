import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from newsapi import NewsApiClient
import google.generativeai as genai

# Load environment variables
load_dotenv()
newsapi_api_key = os.getenv('NEWSAPI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

# Configure Google Generative AI
genai.configure(api_key=google_api_key)

# Initialize NewsApiClient
newsapi = NewsApiClient(api_key=newsapi_api_key)

# Streamlit title
st.title("Stock Price Predictor and Market Sentiment Analysis App1")

# User input for stock ID
stock = st.text_input("Enter the Stock ID", "GOOG")

# Define date range
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Download stock data
google_data = yf.download(stock, start, end)

# Load pre-trained model
model = load_model("Latest_stock_price_model.keras")

# Display stock data
st.subheader("Stock Data")
st.write(google_data)

# Function to plot graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

# Display moving averages
st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data, 0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Prepare data for prediction
splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=google_data.index[splitting_len + 100:]
)

# Display original vs predicted values
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data.Close[:splitting_len + 100], ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)

# Define today's date and the date two days ago
today = datetime.now().strftime('%Y-%m-%d')
two_days_ago = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')

# Function to fetch news headlines
def get_news_headlines(company):
    try:
        # Fetch all articles related to the company
        print(f"Fetching news from {two_days_ago} to {today}")
        all_articles = newsapi.get_everything(
            q=company,
            from_param=two_days_ago,
            to=today,
            language='en',
            sort_by='publishedAt'
        )

        # Extract article titles
        articles = all_articles.get('articles', [])
        titles = [article['title'] for article in articles]

        return titles

    except Exception as e:
        st.error(f"Failed to fetch news: {e}")
        return []

# Function to analyze sentiment using Google Generative AI

def analyze_sentiment(titles):
    # Create the prompt with the instructions and the news headlines
    prompt = (
        "Just answer in one number from 1 to 5, nothing else.\n"
        "I will give you some news headlines relating to a company, "
        "and you need to predict what effect the headlines will have on the company's stock price. "
        "Rate it from 1 to 5, where 5 indicates a high chance of increase and 1 indicates a high chance of decrease.\n"
        + "\n".join(titles)
    )

    # Generate content using the specified model and prompt
    response = genai.GenerativeModel(model_name="gemini-1.5-flash").generate_content(
        [prompt]
    )._result

    # Debugging: Print the entire response to understand its structure
    print("Full response:", response.candidates[0].content.parts[0].text)

    # Attempt to extract and clean the sentiment score
    sentiment_score = response.candidates[0].content.parts[0].text.strip()
    print("Cleaned sentiment score:", sentiment_score)

    return sentiment_score

# Display market sentiment analysis
st.subheader("Market Sentiment Analysis")

if st.button('Get Market Sentiment'):
    try:
        # Fetch news and analyze sentiment
        headlines = get_news_headlines(stock)
        
        # Display latest news headlines in an expandable section
        with st.expander("Latest News Headlines"):
            st.write(headlines)
        
        sentiment_score = analyze_sentiment(headlines)
        
        # Determine sentiment description based on score
        if sentiment_score == "1":
            sentiment_description = "Don't buy"
        elif sentiment_score == "2":
            sentiment_description = "Buy at your own risk"
        elif sentiment_score == "3":
            sentiment_description = "Average Conditions"
        elif sentiment_score == "4":
            sentiment_description = "Favorable conditions"
        elif sentiment_score == "5":
            sentiment_description = "Best time to buy"
        else:
            sentiment_description = "Invalid score"
        
        st.write(f"Market Sentiment Score (From 1(worst) to 5(best)):")
        st.subheader(f"{sentiment_score} - {sentiment_description}")
    except Exception as e:
        st.write(f"An error occurred: {e}")
