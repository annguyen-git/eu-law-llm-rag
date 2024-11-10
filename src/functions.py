from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from bs4 import BeautifulSoup
import requests

import os
import shutil
from bs4 import BeautifulSoup
import requests

# Links for each category
categories_urls = {
    "Intellectual Property": [
        "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32001L0029",
        "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32019L0790",
        "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:31996L0009",
        "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016L0943"
    ],
    "Artificial Intelligence": [
        "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689"
    ],
    "Digital Services": [
        "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2056"
    ],
    "Data Protection": [
        "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679"
    ],
    "Competition Law": [
        "https://eur-lex.europa.eu/LexUriServ/LexUriServ.do?uri=CELEX%3A12008E101%3AEN%3AHTML",
        "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX%3A12008E102",
        "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R1925"
    ],
    "Data Governance": [
        "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32019L1024",
        "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R0868"
    ]
}

# Chat template
PROMPT_TEMPLATE = """
You are a legal assistant trained to answer questions based on legal documents and principles.
Use the following legal context to answer the question.
If the context does not provide a clear answer, just say you don't know.

Context:
{context}

---
Question: {question}

Legal Answer (if available):
"""

# Embeddings and chat model
embeddings_ollama = OllamaEmbeddings(model="MXBAI-EMBED-LARGE")
model_ollama = ChatOllama(model="llama3.2", temperature=0)
LOG_FILE_PATH = 'logs/processed_urls.log'

# Clear all data
def clear_vectorstore(vectorstore_path):
    if os.path.exists(vectorstore_path):
        shutil.rmtree(vectorstore_path)  # Removes the entire directory and its contents
        print(f"Vectorstore at '{vectorstore_path}' has been cleared.")
    else:
        print(f"No vectorstore found at '{vectorstore_path}'.")

# ---------------------Component functions---------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Log file path
def check_if_url_processed(url):
    """Check if the URL has already been processed."""
    if not os.path.exists(LOG_FILE_PATH):
        return False

    with open(LOG_FILE_PATH, 'r') as log_file:
        processed_urls = log_file.readlines()

    processed_urls = [line.strip() for line in processed_urls]
    return url in processed_urls

# Write logs
def log_processed_url(url):
    """Log the processed URL to the log file."""
    with open(LOG_FILE_PATH, 'a') as log_file:
        log_file.write(url + '\n')

# ---------------------Main functions---------------------
def store_data_to_vectorstore(url, category):
    if check_if_url_processed(url):
        # print('Data is already in the vectorstore')
        return
    
    try:
        # Get data from HTML
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve the webpage: {url}")
            return
        print("Internet connected")

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style']):
            element.decompose()

        # Extract visible text
        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        full_text = "\n".join(lines)

        print("Data collected")
        vectorstore_path = f"data/vectorstores/{category.replace(' ', '_')}"
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " "]
        )
        chunks = text_splitter.split_text(full_text)

        # Initialize the vectorstore
        try:
            vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings_ollama)
        except Exception as e:
            print(f"Error initializing vectorstore: {e}")
            vectorstore = Chroma.from_texts(texts=[], embedding=embeddings_ollama, persist_directory=vectorstore_path)
        
        # Check and insert new chunks
        new_chunks = []
        for chunk in chunks:
            embedding = embeddings_ollama.embed_query(chunk)  # Use appropriate method for embeddings
            results = vectorstore.similarity_search_by_vector(embedding, k=1)
            
            if not results or results[0].page_content != chunk:
                new_chunks.append(chunk)

        # Add new chunks to the vectorstore
        if new_chunks:
            embeddings = [embeddings_ollama.embed_query(chunk) for chunk in new_chunks]
            vectorstore.add_texts(texts=new_chunks, embeddings=embeddings)
            print(f"Inserted {len(new_chunks)} new chunks.")
        else:
            print("No new data to insert.")

        # Log the URL as processed
        log_processed_url(url)
        print(f"URL {url} has been logged as processed.")

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")

# ---------------------Input---------------------
def select_category():
    print("Please select a category to process:")
    categories = list(categories_urls.keys())
    
    for i, category in enumerate(categories, start=1):
        print(f"{i}. {category}")
    
    choice = int(input("\nEnter the number of the category: "))
    
    if 1 <= choice <= len(categories):
        selected_category = categories[choice - 1]
        print(f"You selected: {selected_category}")
        return selected_category
    else:
        print("Invalid choice, please try again.")
        return select_category()
    
# Full code
def retrieve_answer():
    selected_category = select_category()
    print(f"Processing category: {selected_category}")
    urls = categories_urls[selected_category]
    for url in urls:
        store_data_to_vectorstore(url, selected_category)

    question = input("Enter your question: ")
    vectorstore_path = f"data/vectorstores/{selected_category.replace(' ', '_')}"
    print(f'Searching in data/vectorstore/{vectorstore_path}')
    try:
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings_ollama)
        retriever = vectorstore.as_retriever(search_type="similarity")
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | model_ollama
        )
        response = rag_chain.invoke(question)
        
        return response
    except Exception as e:
        print(f"Error: {str(e)}")
        return "An error occurred while retrieving data."