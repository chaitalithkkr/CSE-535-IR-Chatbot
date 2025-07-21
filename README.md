# Wikipedia Q/A Chatbot

This project is a Wiki Q/A Chatbot that combines interactive chit-chat with topic-specific information retrieval. It is built using a dataset of over 50,000 Wikipedia documents and provides concise, relevant answers across topics like health, technology, politics, and more.

## Features
- Chit-chat mode powered by BlenderBot for natural conversations.
- Topic classification using zero-shot learning to ensure accurate responses.
- Document retrieval and summarization with TF-IDF for efficient Q/A.
- Exception handling to maintain smooth user interactions.
- Web-based chat interface with topic selection and conversation control.

## Methodology
- **Data Collection:** Wikipedia scraping for 10 major topics (e.g., health, environment, technology).
- **Processing:** Preprocessing and cleaning of scraped documents.
- **Chit-chat Component:** Integrated BlenderBot for engaging, dynamic conversations.
- **Q/A Retrieval:** TF-IDF vectorization and summarization for precise answers.
- **Topic Analysis:** Zero-shot classification using NLP models.

## Tech Stack
- **Languages:** Python
- **Libraries:** BlenderBot, TF-IDF, BERT-based zero-shot models
- **Frameworks:** Flask (web interface)
- **Data Source:** Wikipedia (via `wikipedia` library)

## Conclusion
The chatbot integrates conversational AI with information retrieval, providing both casual and topic-specific interactions. It is designed for accuracy, efficiency, and a smooth user experience.
