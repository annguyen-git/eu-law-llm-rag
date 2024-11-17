# Lawyer Chatbot

This project processes legal documents related to data such as data governmance, data proctection..., across various categories, storing them in a vector store. It uses Langchain, Chroma, and Ollama models to analyze and retrieve answers to legal questions based on the content of these documents.

## Structure

![alttext](resources/llm_rag.jpg)

### Data Flow
- User selects topic, each topic is mapped to an url.
- The logs contain processed url, if the url is new, text is extracted from HTML pages for each topic. I the url is in logs, skip embedding.
- The extracted text is processed and compared with existing data in the vector database. If the text is new, it is embedded and stored in the database.
- The user's question is embedded and used for similarity search within the vector database. Relevant context is retrieved and combined with a chat template to create the final prompt.
- The LLM model generates a response based on the final prompt.

### Embeddings & Chat Model
Embeddings: The MXBAI-EMBED-LARGE model is used to generate embeddings for each text chunk.

Chat Model: The llama3.2 model is used to generate answers based on the retrieved context.

### Categories
The following categories are available for querying:

- Intellectual Property
- Artificial Intelligence
- Digital Services
- Data Protection
- Competition Law
- Data Governance

Each category has associated legal URLs for scraping and processing.

## Prerequisites
- Install needed packages
```bash
pip install -r requirements.txt
```
- Install Ollama model Llama 3.2 3B [here](https://github.com/ollama/ollama)
## How to use

![alttext](resources/llm_rag.gif)

- Run this command
```bash
python src/main.py
```

- Select the desired category by entering the corresponding number on the list.
- Ask a Question.
- Output: The system will output the answer based on the most relevant legal documents found in the vector store.
- Option: type 'documents' for original documents.

## Things to be added
- Web app interface.
- Model fine tunning.
- Answer log for repeated questions.