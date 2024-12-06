#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify, render_template
import json
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM
from flask_cors import CORS
from collections import defaultdict  # Add this line
import time  

# Initialize Flask app
app = Flask(__name__)


# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load the query classifier
zero_shot_classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
query_labels = ["chitchat", "wiki_query"]
query_hypothesis_template = "This is a {}."

# Load the chitchat model
blenderbot_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
blenderbot_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

# Load the T5 model for summarization
t5_model_name = "t5-base"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# Reload the zero-shot classification model and tokenizer for topic classification
load_directory = "./saved_zero_shot_model"
tokenizer = AutoTokenizer.from_pretrained(load_directory)
model = AutoModelForSequenceClassification.from_pretrained(load_directory)
pipe = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# Define topics and hypothesis template for topic classification
topics = [
    "Health",
    "Environment",
    "Technology",
    "Economy",
    "Entertainment",
    "Sports",
    "Politics",
    "Education",
    "Travel",
    "Food",
]
topic_hypothesis_template = "This query is related to {}."
leaving_remarks = ["bye", "goodbye", "exit", "see you", "later", "quit"]  # Define leaving remarks

# Utility functions
def classify_query_type(query, classifier, labels, threshold=0.5):
    output = classifier(query, labels, hypothesis_template=query_hypothesis_template)
    scores = output["scores"]
    best_label = output["labels"][0]
    best_score = scores[0]
    return {"label": best_label, "score": best_score} if best_score > threshold else {"label": "uncertain", "score": best_score}

def chat_with_blenderbot(input_text):
    inputs = blenderbot_tokenizer(input_text, return_tensors="pt")
    outputs = blenderbot_model.generate(**inputs)
    return blenderbot_tokenizer.decode(outputs[0], skip_special_tokens=True)

def classify_multi_topics(query, pipe, topics, threshold=0.1, top_n=5):
    output = pipe(query, topics, hypothesis_template=topic_hypothesis_template)
    labels = output["labels"]
    scores = output["scores"]
    return [label for label, score in zip(labels, scores) if score > threshold][:top_n]

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    filtered_words = [
        lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words
    ]
    return " ".join(filtered_words)

def load_preprocessed_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def create_tfidf_vectorizer(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

def get_most_relevant_articles(query, tfidf_vectorizer, tfidf_matrix, articles, top_n=3):
    query_vec = tfidf_vectorizer.transform([query])
    cosine_similarities = np.dot(tfidf_matrix, query_vec.T).toarray().flatten()
    sorted_indices = cosine_similarities.argsort()[::-1]
    unique_articles = []
    seen_urls = set()
    for idx in sorted_indices:
        if len(unique_articles) >= top_n:
            break
        article = articles[idx]
        article_url = article.get('url', None)
        if article_url and article_url not in seen_urls:
            unique_articles.append(article)
            seen_urls.add(article_url)
    return unique_articles

def generate_meaningful_summary_t5(combined_text, max_length=800, min_length=100):
    inputs = t5_tokenizer.encode("summarize: " + combined_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = t5_model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=1.0,
        num_beams=4,
        early_stopping=True
    )
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def combine_summaries(articles):
    combined_text = " ".join([
        article['summary']['text_en'] if 'summary' in article and 'text_en' in article['summary'] else "No Summary"
        for article in articles
    ])
    if not combined_text.strip():
        return "No valid summaries available to generate a meaningful summary."
    return generate_meaningful_summary_t5(combined_text)

def wiki_qa_system(query, preprocessed_data, selected_topics=None, top_n=3):
    # Use selected_topics if provided; otherwise, classify the query
    relevant_topics = selected_topics if selected_topics else classify_multi_topics(query, pipe, topics, threshold=0.1, top_n=5)

    all_articles = []
    for topic, data in preprocessed_data.items():
        if topic in relevant_topics and 'articles' in data and isinstance(data['articles'], list):
            for article in data['articles']:
                if isinstance(article, dict) and 'preprocessed_summary' in article:
                    article['topic'] = topic
                    all_articles.append(article)
    if not all_articles:
        raise ValueError("No valid articles found for the relevant topics.")

    articles_summaries = [
        preprocess_text(article['preprocessed_summary']) for article in all_articles
        if 'preprocessed_summary' in article
    ]
    if not articles_summaries:
        raise ValueError("No preprocessed summaries found for matching.")

    tfidf_vectorizer, tfidf_matrix = create_tfidf_vectorizer(articles_summaries)
    most_relevant_articles = get_most_relevant_articles(query, tfidf_vectorizer, tfidf_matrix, all_articles, top_n)
    combined_summary = combine_summaries(most_relevant_articles)
    
    answers = [{
        'title': article.get('title', "No Title"),
        'topic': article.get('topic', "No Topic"),
        'url': article.get('url', "No URL")
    } for article in most_relevant_articles]

    return {
        'relevant_topics': relevant_topics,
        'combined_summary': combined_summary,
        'answers': answers
    }


# Load preprocessed data
preprocessed_data = load_preprocessed_data("preprocessed_data.json")

CONVERSATION_FILE = "conversations.json"
if not os.path.exists(CONVERSATION_FILE):
    with open(CONVERSATION_FILE, "w") as f:
        json.dump([], f)

def save_conversation(user_query, response, topics, response_time):
    new_entry = {
        "user_query": user_query,
        "response": response,
        "topics": topics,
        "timestamp": datetime.utcnow().isoformat()  # Add timestamp in ISO format
    }
    if response_time is not None:
        new_entry["response_time"] = response_time

    # Append new entry to the JSON file
    with open(CONVERSATION_FILE, "r+") as f:
        data = json.load(f)
        data.append(new_entry)
        f.seek(0)
        json.dump(data, f)
        
@app.route('/visualization_data', methods=['GET'])
def visualization_data():
    try:
        with open(CONVERSATION_FILE, "r") as f:
            data = json.load(f)
        topic_counts = defaultdict(int)
        for entry in data:
            for topic in entry.get("topics", []):
                topic_counts[topic] += 1
        print("Topic Counts:", topic_counts)  # Debugging
        response = {
            "total_conversations": len(data),
            "topic_counts": dict(topic_counts)
        }
        return jsonify(response)
    except Exception as e:
        print("Error in visualization_data:", e)
        return jsonify({"error": "Unable to process visualization data"}), 500



@app.route('/visualization')
def visualization():
    return render_template('visualizations.html')
    
@app.route('/time_between_queries_data', methods=['GET'])
def time_between_queries_data():
    try:
        with open(CONVERSATION_FILE, "r") as f:
            data = json.load(f)

        # Extract timestamps and ensure they are valid
        timestamps = [
            datetime.fromisoformat(entry["timestamp"])  # Parse ISO format timestamps
            for entry in data if "timestamp" in entry
        ]
        timestamps.sort()  # Ensure timestamps are in chronological order

        # Calculate time gaps between consecutive queries
        time_gaps = [
            (timestamps[i] - timestamps[i - 1]).total_seconds()
            for i in range(1, len(timestamps))
        ]

        return jsonify({"time_gaps": time_gaps})
    except Exception as e:
        print(f"Error processing time between queries data: {e}")
        return jsonify({"error": "Could not process time between queries data"}), 500


@app.route('/')
def home():
    return render_template('Index.html')


@app.route('/response_time_data', methods=['GET'])
def response_time_data():
    with open(CONVERSATION_FILE, "r") as f:
        data = json.load(f)
    response_times = [entry["response_time"] for entry in data if "response_time" in entry]
    return jsonify({"response_times": response_times})

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("query", "").strip().lower()
    selected_topics = request.json.get("topics", None)  # Get topics from the request
    start_time = time.time()

    # Check if "chitchat" is in the selected topics and send directly to chitchat model
    if selected_topics and "chitchat" in selected_topics:
        response = chat_with_blenderbot(user_input)
        response_time = time.time() - start_time
        save_conversation(user_input, response, ["chitchat"], response_time)
        return jsonify({"response": response})

    # Check for leaving remarks
    if any(remark in user_input for remark in leaving_remarks):
        response = "Goodbye! Have a great day!"
        save_conversation(user_input, response, [], None)
        return jsonify({"response": response})

    try:
        # If topics are provided by the user, bypass classification
        if selected_topics:
            relevant_topics = selected_topics
        else:
            query_type = classify_query_type(user_input, zero_shot_classifier, query_labels, threshold=0.5)
            if query_type["label"] == "chitchat":
                response = chat_with_blenderbot(user_input)
                response_time = time.time() - start_time
                save_conversation(user_input, response, ["chitchat"], response_time)
                return jsonify({"response": response})

            elif query_type["label"] == "wiki_query":
                relevant_topics = classify_multi_topics(user_input, pipe, topics, threshold=0.1, top_n=5)
            else:
                response = "The query could not be classified confidently. Please try again."
                response_time = time.time() - start_time
                save_conversation(user_input, response, [], response_time)
                return jsonify({"response": response})

        # Retrieve documents directly based on the relevant topics
        result = wiki_qa_system(user_input, preprocessed_data, selected_topics=relevant_topics, top_n=3)
        topics_list = "\n- " + "\n- ".join(relevant_topics)
        summary = result["combined_summary"]
        articles = "\n".join(f"- [{article['title']}]({article['url']})" for article in result["answers"])

        formatted_response = (
            f"Classified Topics:\n{topics_list}\n\n"
            f"Summary:\n{summary}\n\n"
            f"Articles:\n{articles}"
        )

        response_time = time.time() - start_time
        save_conversation(user_input, formatted_response, relevant_topics, response_time)
        return jsonify({"response": formatted_response})

    except Exception as e:
        response_time = time.time() - start_time
        save_conversation(user_input, str(e), [], response_time)
        return jsonify({"error": "Failed to process your query."})


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:
