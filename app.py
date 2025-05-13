import gradio as gr
import pandas as pd
from media_trust import query, process_data, analyse_sentiment, add_bias_annotation, set_article_extremity, add_article_summaries

def process_news(topic):
    raw_df = query(topic)
    processed_df = process_data(raw_df)
    sentiment_df = analyse_sentiment(processed_df)
    bias_df = add_bias_annotation(sentiment_df)
    extremity_df = set_article_extremity(bias_df)
    final_df = add_article_summaries(extremity_df)
    return final_df[['title', 'summary', 'bias_score', 'extremity_pct', 'source']]

with gr.Blocks() as interface:
    with gr.Column():
        topic_input = gr.Textbox(label="Enter a topic", placeholder="e.g., Tesla")
        output_table = gr.DataFrame(headers=["Title", "Summary", "Bias", "Extremity %", "Source"], interactive=False)
        topic_input.submit(process_news, inputs=topic_input, outputs=output_table)

interface.launch()
