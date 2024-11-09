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

url = 'https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32001L0029'

PROMPT_TEMPLATE = """
You are a legal assistant trained to answer questions based on legal documents and principles.
Use the following legal context to answer the question.
If the context does not provide a clear answer, respond with "Sorry. I can't find the answer. Please check the topic again."

Context:
{context}

---
Question: {question}

Legal Answer (if available):
"""

# Embeddings and chat model
embeddings_ollama = OllamaEmbeddings(model="mxbai-embed-large")
model_ollama = ChatOllama(model="llama3.2", temperature=0)
vectorstore_path='data/vetorstore/chromadb'

# get data from html and store embedded data to chromadb
def store_data_to_vectorstore(url):
    # get data from html
    response = requests.get(url)
    if response.status_code == 200:
        print('Internet connected')
        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = soup.find('div', {'id': 'TexteOnly'}) 

        if main_content:
            all_text = []
            paragraphs = main_content.find_all('p')
            for i, p in enumerate(paragraphs):
                if i == 0:
                    continue
                paragraph_text = p.get_text(strip=True)
                all_text.append(paragraph_text)
            full_text = "\n".join(all_text)
            print('Data collected')
        else:
            print("Main content not found.")

    # separate text to chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " "]
    )
    chunks = text_splitter.split_text(full_text)

    # write to chromadb
    try:
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings_ollama)
    except:
        vectorstore = Chroma.from_texts(texts=[], embedding=embeddings_ollama, persist_directory=vectorstore_path)
    new_chunks = []
    for chunk in chunks:
        embedding = embeddings_ollama.embed_query(chunk)
        results = vectorstore.similarity_search_by_vector(embedding, k=1)
        if not results or results[0].page_content != chunk:
            new_chunks.append(chunk)

    if new_chunks:
        embeddings = [embeddings_ollama.embed_query(chunk) for chunk in new_chunks]
        vectorstore.add_texts(texts=new_chunks, embeddings=embeddings)
        print(f"Inserted {len(new_chunks)} new chunks.")
    else:
        print("No new data to insert.")


def retrieve_answer(question):
    try:
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings_ollama)
        retriever = vectorstore.as_retriever(search_type="similarity")
        relevant_chunks = retriever.invoke("who will provide adequate legal protection against the manufature of devices which have purpose of bypassing a technological protection measure.")
        print (relevant_chunks)
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
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



