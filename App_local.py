#!pip install transformers

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForSeq2SeqLM, 
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    pipeline
)
from flask import Flask, request, jsonify, render_template
import nltk
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize the Flask app
app = Flask(__name__)

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load the query classifier
tokenizer = AutoTokenizer.from_pretrained("valhalla/distilbart-mnli-12-1")
model = AutoModelForSequenceClassification.from_pretrained("valhalla/distilbart-mnli-12-1")
zero_shot_classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

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
topic_tokenizer = AutoTokenizer.from_pretrained(load_directory)
topic_model = AutoModelForSequenceClassification.from_pretrained(load_directory)
pipe = pipeline("zero-shot-classification", model=topic_model, tokenizer=topic_tokenizer)

# Define topics and hypothesis template for topic classification
topics = [
    "Health", "Environment", "Technology", "Economy", "Entertainment", 
    "Sports", "Politics", "Education", "Travel", "Food",
]
topic_hypothesis_template = "This query is related to {}."

# Define leaving remarks
leaving_remarks = ["bye", "goodbye", "exit", "see you", "later", "quit"]

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

def wiki_qa_system(query, preprocessed_data, top_n=3):
    relevant_topics = classify_multi_topics(query, pipe, topics, threshold=0.1, top_n=5)
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

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("query", "").strip().lower()

    # Check if the user input contains leaving remarks
    if any(remark in user_input for remark in leaving_remarks):
        return jsonify({"response": "Goodbye! Have a great day!"})

    # Classify the query type (chitchat or wiki_query)
    query_type = classify_query_type(user_input, zero_shot_classifier, query_labels, threshold=0.5)

    # Handle chitchat response
    if query_type["label"] == "chitchat":
        response = chat_with_blenderbot(user_input)
        return jsonify({"response": response})

    # Handle wiki_query response
    elif query_type["label"] == "wiki_query":
        try:
            result = wiki_qa_system(user_input, preprocessed_data, top_n=3)
            return jsonify({
                "relevant_topics": result["relevant_topics"],
                "summary": result["combined_summary"],
                "articles": result["answers"]
            })
        except ValueError as ve:
            return jsonify({"error": str(ve)})

    # If the query cannot be classified
    else:
        return jsonify({"response": "The query could not be classified confidently. Please try again."})


if __name__ == "__main__":
    app.run(debug=True)
