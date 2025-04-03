import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='data/fetch_news.log'
)
logger = logging.getLogger(__name__)

# Get RapidAPI key from environment variable
RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY')
if not RAPIDAPI_KEY:
    raise ValueError("RAPIDAPI_KEY environment variable is not set. Please set it before running this script.")

# Set your RapidAPI credentials
RAPIDAPI_HOST = "news-api14.p.rapidapi.com"

def fetch_news_for_ticker(ticker, days=7):
    """Fetch news articles for a given ticker."""
    try:
        url = "https://news-api14.p.rapidapi.com/search"
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        querystring = {
            "q": ticker,
            "language": "en",
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "limit": "50"
        }

        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": RAPIDAPI_HOST
        }

        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        
        data = response.json()
        articles = data.get('articles', [])
        
        # Process articles
        news_data = []
        for article in articles:
            news_data.append({
                'ticker': ticker,
                'date': article.get('publishedDate', ''),
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'source': article.get('source', {}).get('name', '')
            })
        
        return pd.DataFrame(news_data)
        
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {str(e)}")
        return pd.DataFrame()

def main():
    # List of tickers to fetch news for
    tickers = ['NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'MU', 'MRVL', 'ASML', 'TSM']
    
    all_news = []
    for ticker in tickers:
        logger.info(f"Fetching news for {ticker}")
        news_df = fetch_news_for_ticker(ticker)
        if not news_df.empty:
            all_news.append(news_df)
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    if all_news:
        final_df = pd.concat(all_news, ignore_index=True)
        final_df.to_csv('data/stock_news_data.csv', index=False)
        logger.info("News data saved to data/stock_news_data.csv")
    else:
        logger.warning("No news data was fetched")

if __name__ == "__main__":
    main() 