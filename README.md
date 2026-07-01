# PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about a PDF document using semantic search and Google's Gemini model.

## Features

- Load PDF documents
- Split documents into chunks
- Generate embeddings using Hugging Face
- Store embeddings in FAISS
- Retrieve the most relevant document chunks
- Generate answers using Gemini

## Tech Stack

- Python
- LangChain
- Hugging Face Embeddings
- FAISS
- Google Gemini
- PyPDF

## Project Structure

```
pdf-rag-chatbot/
│
├── app.py
├── build_index.py
├── requirements.txt
├── .gitignore
├── .env.example
├── data/
│   └── ml1.pdf
└── vector_store/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/kartik2245/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key
```

Build the vector database:

```bash
python build_index.py
```

Run the chatbot:

```bash
python app.py
```

## Example

```
Ask a question:
> What is Feature Scaling?

Answer:
Feature Scaling is the process of transforming numerical features to a similar scale...
```

## Future Improvements

- Upload custom PDFs
- Chat history
- Streamlit web interface
- Multiple PDF support
- Hybrid Search