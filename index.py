from flask import Flask, request, jsonify
import pandas as pd
from media_trust import (
    query, process_data, analyse_sentiment,
    add_bias_annotation, set_article_extremity, add_article_summaries
)

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "MediaTrust API is live"})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    topic = data.get("topic")

    if not topic:
        return jsonify({"error": "Missing 'topic' in request"}), 400

    try:
        raw_df = query(topic)
        if raw_df is None or raw_df.empty:
            return jsonify({"message": "No articles found"}), 200

        processed_df = process_data(raw_df)
        sentiment_df = analyse_sentiment(processed_df)
        bias_df = add_bias_annotation(sentiment_df)
        extreme_df = set_article_extremity(bias_df)
        final_df = add_article_summaries(extreme_df)

        output = final_df[[
            'title', 'summary', 'sentiment_label',
            'bias_label', 'extremity_pct', 'url', 'publishedAt'
        ]].to_dict(orient='records')

        return jsonify({"results": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
