import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from textblob import TextBlob
import numpy as np
from datetime import datetime
import spacy
from tqdm import tqdm
import gc
from collections import defaultdict
import json
import logging

# Import configuration settings
from M_config import MODEL_NAME, TOPICS, SENTIMENT_THRESHOLD, LOG_FILE_PATH, SAVE_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=LOG_FILE_PATH
)
logger = logging.getLogger(__name__)

# Memory optimization for Apple M2
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('medium')
torch.set_num_threads(6)

# Load FinBERT tokenizer & model
logger.info("Loading FinBERT model...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

# Load spaCy for NER and topic extraction
logger.info("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# Initialize keyword tracking
keyword_stats = {
    topic: {keyword: {'count': 0, 'total_sentiment': 0.0, 'sentences': []} 
            for keyword in keywords}
    for topic, keywords in TOPICS.items()
}

def get_finbert_sentiment(text):
    """Get sentiment score using FinBERT."""
    if pd.isna(text) or text.strip() == "":
        return 0.0

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        del inputs
    
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
    del outputs
    gc.collect()
    
    return scores[2] - scores[0]  # Positive - Negative

def analyze_topic_sentiment(doc, text):
    """Analyze sentiment for specific topics in the text."""
    topic_sentiments = {}
    
    for topic, keywords in TOPICS.items():
        topic_sentiment = 0.0
        topic_matches = []
        
        # Find keyword matches in the text
        for keyword in keywords:
            # Look for exact matches of keywords
            matches = [token for token in doc if token.text.lower() == keyword.lower()]
            if matches:
                # Get sentiment for sentences containing the keyword
                sentences = [sent.text for sent in doc.sents if any(keyword.lower() in sent.text.lower() for keyword in keywords)]
                if sentences:
                    # Calculate sentiment for each sentence
                    sentiments = [get_finbert_sentiment(sent) for sent in sentences]
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    
                    # Update keyword statistics
                    keyword_stats[topic][keyword]['count'] += len(matches)
                    keyword_stats[topic][keyword]['total_sentiment'] += avg_sentiment
                    
                    # Store sample sentences (up to 3 per keyword)
                    current_samples = keyword_stats[topic][keyword]['sample_sentences']
                    new_samples = [sent for sent in sentences if sent not in current_samples]
                    keyword_stats[topic][keyword]['sample_sentences'].extend(new_samples[:3])
                    
                    topic_sentiment += avg_sentiment
                    topic_matches.extend(matches)
        
        if topic_matches:
            topic_sentiments[topic] = topic_sentiment / len(topic_matches)
        else:
            topic_sentiments[topic] = 0.0
    
    return topic_sentiments

def analyze_keyword_effectiveness():
    """Analyze which keywords are most effective at capturing sentiment."""
    effective_keywords = defaultdict(list)
    
    print("\nKeyword Effectiveness Analysis:")
    print("==============================\n")
    
    for topic, keywords in keyword_stats.items():
        print(f"Topic: {topic}")
        print("-" * (len(topic) + 7) + "\n")
        
        for keyword, stats in keywords.items():
            if stats['count'] > 0:
                avg_sentiment = stats['total_sentiment'] / stats['count']
                effectiveness_score = abs(avg_sentiment) * stats['count']
                
                print(f"Keyword: {keyword}")
                print(f"  Occurrences: {stats['count']}")
                print(f"  Average Sentiment: {avg_sentiment:.3f}")
                print(f"  Effectiveness Score: {effectiveness_score:.3f}")
                
                if stats['sample_sentences']:  # Changed from 'sentences' to 'sample_sentences'
                    print("  Sample Sentences:")
                    for sentence in stats['sample_sentences'][:3]:  # Show up to 3 sample sentences
                        print(f"    - {sentence}")
                print()
                
                # Consider a keyword effective if it appears at least 5 times
                # and has an average absolute sentiment score > 0.1
                if stats['count'] >= 5 and abs(avg_sentiment) > 0.1:
                    effective_keywords[topic].append(keyword)
        print()
    
    # Save the results to a JSON file
    with open(f"{SAVE_PATH}keyword_effectiveness.json", "w") as f:
        json.dump(dict(keyword_stats), f, indent=2)
    
    return effective_keywords

def process_news_batch(batch):
    """Process a batch of news articles and return sentiment scores."""
    daily_sentiments = defaultdict(list)
    daily_topic_sentiments = defaultdict(list)
    
    for _, row in batch.iterrows():
        date = row['Date']
        stock = row['stock']  # Changed from 'Stock' to 'stock'
        # Combine title and description for analysis
        text = f"{row['title']} {row['description']}"
        
        # Get general sentiment
        sentiment_score = get_finbert_sentiment(text)
        daily_sentiments[(date, stock)].append(sentiment_score)
        
        # Get topic-specific sentiments
        doc = nlp(text.lower())
        analyzed_topics = analyze_topic_sentiment(doc, text)
        
        for topic, score in analyzed_topics.items():
            if score != 0:  # Only store non-zero sentiment scores
                daily_topic_sentiments[(date, stock, topic)].append(score)
    
    return daily_sentiments, daily_topic_sentiments

def aggregate_sentiments(daily_sentiments, daily_topic_sentiments):
    """Aggregate daily sentiment scores."""
    general_records = []
    topic_records = []
    
    # Aggregate general sentiments
    for (date, stock), scores in daily_sentiments.items():
        if scores:  # Only process if we have scores
            avg_score = sum(scores) / len(scores)
            general_records.append({
                'Date': date,
                'Stock': stock,
                'General Sentiment Score': avg_score
            })
    
    # Aggregate topic sentiments
    for (date, stock, topic), scores in daily_topic_sentiments.items():
        if scores:  # Only process if we have scores
            avg_score = sum(scores) / len(scores)
            topic_records.append({
                'Date': date,
                'Stock': stock,
                'Topic': topic,
                'Topic Sentiment Score': avg_score
            })
    
    return general_records, topic_records

# Main execution
print("Loading FinBERT model...")
finbert = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
finbert.eval()  # Set the model to evaluation mode

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

print("Loading news data...")
df_news = pd.read_csv("data/stock_news_data.csv")
df_news['Date'] = pd.to_datetime(df_news['Date'])

# Initialize keyword tracking
keyword_stats = defaultdict(lambda: defaultdict(lambda: {
    'count': 0,
    'total_sentiment': 0.0,
    'sample_sentences': []
}))

# Process news in batches
print("Analyzing sentiments...")
BATCH_SIZE = 100  # Increased batch size for better performance
all_daily_sentiments = defaultdict(list)
all_daily_topic_sentiments = defaultdict(list)

for i in tqdm(range(0, len(df_news), BATCH_SIZE)):
    batch = df_news.iloc[i:i+BATCH_SIZE]
    daily_sentiments, daily_topic_sentiments = process_news_batch(batch)
    
    # Merge batch results
    for key, values in daily_sentiments.items():
        all_daily_sentiments[key].extend(values)
    for key, values in daily_topic_sentiments.items():
        all_daily_topic_sentiments[key].extend(values)
    
    gc.collect()

# Analyze keyword effectiveness before aggregating results
print("\nAnalyzing keyword effectiveness...")
effective_keywords = analyze_keyword_effectiveness()

# Aggregate results
print("\nAggregating results...")
general_records, topic_records = aggregate_sentiments(all_daily_sentiments, all_daily_topic_sentiments)

# Convert to DataFrames
df_general = pd.DataFrame(general_records)
df_topic = pd.DataFrame(topic_records)

# Ensure all stock-date-topic combinations exist
print("Ensuring complete topic coverage...")
all_dates = df_general['Date'].unique()
all_stocks = df_general['Stock'].unique()
all_topics = list(TOPICS.keys())

complete_topic_records = []
for date in all_dates:
    for stock in all_stocks:
        for topic in all_topics:
            existing = df_topic[
                (df_topic['Date'] == date) & 
                (df_topic['Stock'] == stock) & 
                (df_topic['Topic'] == topic)
            ]
            
            if len(existing) == 0:
                complete_topic_records.append({
                    'Date': date,
                    'Stock': stock,
                    'Topic': topic,
                    'Topic Sentiment Score': 0.0  # Default score for missing topics
                })

# Add missing records and sort
if complete_topic_records:
    df_topic = pd.concat([
        df_topic,
        pd.DataFrame(complete_topic_records)
    ], ignore_index=True)

# Sort DataFrames
df_general = df_general.sort_values(['Date', 'Stock'])
df_topic = df_topic.sort_values(['Date', 'Stock', 'Topic'])

# Save results
print("Saving results...")
df_general.to_csv("data/general_sentiment.csv", index=False)
df_topic.to_csv("data/topic_sentiment.csv", index=False)

print("\nSuggested Topic Keywords Updates:")
print("================================")
for topic, keywords in effective_keywords.items():
    if keywords:
        print(f"\n{topic}:")
        print(f"Current keywords: {TOPICS[topic]}")
        print(f"Suggested keywords (based on effectiveness): {keywords}")
    else:
        print(f"\n{topic}:")
        print("Warning: No effective keywords found. Consider reviewing all keywords for this topic.")

# Print summary statistics
print(f"""
Sentiment Analysis Summary:
-------------------------
General Sentiment Statistics:
- Total days analyzed: {df_general['Date'].nunique()}
- Total stocks covered: {df_general['Stock'].nunique()}
- Average sentiment score: {df_general['General Sentiment Score'].mean():.3f}

Topic Sentiment Statistics:
- Total topics tracked: {df_topic['Topic'].nunique()}
- Total topic-stock-day combinations: {len(df_topic)}
- Average topic sentiment score: {df_topic['Topic Sentiment Score'].mean():.3f}

Most Positive Topics:
{df_topic.groupby('Topic')['Topic Sentiment Score'].mean().sort_values(ascending=False).head().to_string()}

Most Negative Topics:
{df_topic.groupby('Topic')['Topic Sentiment Score'].mean().sort_values().head().to_string()}

Files saved:
- general_sentiment.csv: Daily stock sentiment scores
- topic_sentiment.csv: Daily topic-specific sentiment scores
- keyword_effectiveness.json: Detailed keyword analysis
""") 