import requests
import pandas as pd
import gradio as gr
import datetime
import nltk
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

from transformers import pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError("API_KEY not found. Make sure to set it in the .env file.")

SOURCE_BIAS_MAP = {
    "fox news": "right",
    "breitbart": "right",
    "new york post": "right",
    "the wall street journal": "center-right",
    "reuters": "center",
    "associated press": "center",
    "bloomberg": "center",
    "npr": "center-left",
    "cnn": "left",
    "msnbc": "left",
    "the new york times": "left",
    "the washington post": "left",
    "the guardian": "left",
    "bbc news": "center",
    "sky news": "center-right",
    "the telegraph": "right",
    "the times": "center-right",
    "daily mail": "right",
    "the independent": "center-left",
    "the sun": "right",
    "financial times": "center",
}

BIAS_SCORE_MAP = {
    "left": -1,
    "center-left": -0.5,
    "center": 0,
    "center-right": 0.5,
    "right": 1,
    "unknown": 0
}

def query(query, sort_by="popularity", max_tokens=100):

    if query == "":
        print("Topic needs to be passed in")
        return
    
    today = datetime.today()
    seven_days_ago = today - timedelta(days=20)
    from_date = seven_days_ago.strftime('%Y-%m-%d')
    to_date = today.strftime('%Y-%m-%d')
    
    base_url = "https://newsapi.org/v2/everything"
    url = f"{base_url}?q={query}&from={from_date}&to={to_date}&sortBy={sort_by}&apiKey={api_key}"
    news = None

    try:
        news_response = requests.get(url, timeout=10)
        if news_response.status_code == 200:
            news = news_response.json()

        else:
            print("API error has occured", news_response.status_code)
    except Exception:
        print('An exception occurred')

    article_arr = news["articles"]
    extracted_data = []

    for article in article_arr:
        extracted_data.append({
            "title": article.get("title", "N/A"),
            "description": article.get("description", "N/A"),
            "source_name": article.get("source", {}).get("name", "N/A"),
            "url": article.get("url", "N/A"),
            "publishedAt": article.get("publishedAt", "N/A")
        })

    df = pd.DataFrame(extracted_data)
    return df


def process_data(df):
    df_cleaned = df.dropna(subset=["title", "description"])
    df_cleaned = df_cleaned[df_cleaned["title"].str.strip() != ""]
    df_cleaned = df_cleaned[df_cleaned["description"].str.strip() != ""]
    df_cleaned = df_cleaned.drop_duplicates(subset=["title", "url"])
    df_cleaned["text"] = df_cleaned["title"] + df_cleaned["description"].str.lower()
    return df_cleaned

def analyse_sentiment(df):
    
    analyser = SentimentIntensityAnalyzer()
    
    df['compound'] = [analyser.polarity_scores(x)['compound'] for x in df['text']]
    df['neg'] = [analyser.polarity_scores(x)['neg'] for x in df['text']]
    df['neu'] = [analyser.polarity_scores(x)['neu'] for x in df['text']]
    df['pos'] = [analyser.polarity_scores(x)['pos'] for x in df['text']]
    
    def label_sentiment(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    df['sentiment_label'] = df['compound'].apply(label_sentiment)
    return df

def get_bias_label(source_name):
        source = source_name.strip().lower()
        return SOURCE_BIAS_MAP.get(source, "unknown")

def add_bias_annotation(df):
    df['bias_label'] = df['source_name'].apply(get_bias_label)
    return df

def set_article_extremity(df, top_n=5):
    def get_bias_extremity(label):
        return BIAS_SCORE_MAP.get(label, 0)

    df['bias_score'] = df['bias_label'].apply(get_bias_extremity)

    df['extremity_score'] = df['compound'].abs() + df['bias_score'].abs()

    df['extremity_pct'] = (df['extremity_score'] / 2) * 100
    df['extremity_pct'] = df['extremity_pct'].round(1)

    df = df.sort_values(by='extremity_score', ascending=False)
    df['extreme'] = False
    df.loc[df.index[:top_n], 'extreme'] = True

    return df

def summarise_text(row, max_tokens=512):
    try:
        text = row['text'] if 'text' in row and pd.notna(row['text']) else ''
        source_name = row['source_name'] if 'source_name' in row and pd.notna(row['source_name']) else 'unknown'

        input_length = len(text.split())

        if input_length < 40:
            max_length = max(10, int(input_length / 2))
        else:
            max_length = min(input_length - 10, max_tokens)
        min_length = max(10, max_length - 10)

        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        summary_text = summary[0]['summary_text']

        bias_label = get_bias_label(source_name)

        return pd.Series({
            'summary': summary_text,
            'bias_score': bias_label,
            'source': source_name
        })

    except Exception as e:
        print(f"Error summarising row: {e}")
        return pd.Series({
            'summary': 'Summary unavailable',
            'bias_score': 'unknown',
            'source': 'unknown'
        })

def add_article_summaries(df, max_tokens=512):
    summary_df = df.apply(summarise_text, axis=1, max_tokens=max_tokens)
    df[['summary', 'bias_score', 'source']] = summary_df
    return df

def main():
    raw_df = query("Tesla")
    processed_df = process_data(raw_df)
    sentiment_df = analyse_sentiment(processed_df)
    bias_df = add_bias_annotation(sentiment_df)
    extremity_df = set_article_extremity(bias_df)
    final_df = add_article_summaries(extremity_df)

if __name__ == "__main__":
    main()
