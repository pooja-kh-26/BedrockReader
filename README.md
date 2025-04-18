
# 📄 Bedrock Reader - A RAG System for PDF-Based Question Answering

Retrieval-Augmented Generation system that answers document-based queries using AWS Bedrock, LangChain, LLaMA, and FAISS.



## 📖 Introduction

Bedrock Reader is a simple yet powerful Retrieval-Augmented Generation (RAG) application built for answering user queries based on PDF documents. In this project, we’ve used a Karnataka Travel Guide as the knowledge base. The system reads PDFs, processes them into searchable vector embeddings, retrieves relevant chunks based on a user prompt, and generates context-aware responses using LLaMA via AWS Bedrock.




## Demo 
https://github.com/user-attachments/assets/d2be3975-745c-4e32-b79b-81e01fe68a6e

## 🖥️ Usage

- ####  Open the app:
👉 https://bedrockreader.streamlit.app/

- #### Update Vectors:
Click the “Update Vectors” button — this reads the PDFs, splits them into chunks, generates vector embeddings using LLaMA, and updates the FAISS vector store.

- #### Ask a Question:
Enter a prompt related to the document content (for example: “Best tourist places in Karnataka?”).

- #### Get the Answer:
Click on the “LLaMA” button — this retrieves the most relevant document chunks from FAISS, combines them with your query, and sends it to LLaMA via AWS Bedrock for a smart, document-grounded response.



## 📊 System Architecture
![Image](https://github.com/user-attachments/assets/ffe17d66-2fbb-4b66-a046-2386481d8390)
![Image](https://github.com/user-attachments/assets/344aa82b-a20c-4dfb-b202-2d53ebbf08e3)
![Image](https://github.com/user-attachments/assets/28cfc9e7-e31f-45a0-bb0b-7d5cc688feb6)

## 🔍 Workflow
Here’s a detailed breakdown of how the Bedrock Reader system works:

1️⃣ **Data Ingestion**
- Reads all PDF files from a specified folder.

- Uses LangChain’s PDF loader to extract text content from these files.

2️⃣ **Document Preparation**
#### Chunking:

- Splits the extracted PDF content into smaller, manageable text chunks.

- This helps in precise retrieval and avoids overloading the LLM with too much data.

#### Embedding Generation:

- Uses LLaMA Embedding model (via LangChain) to convert each chunk of text into a dense vector representation.

#### Vector Storage:

- Stores these vector embeddings in a FAISS vector database for efficient similarity-based search.

3️⃣ **Query and Retrieval**
- User enters a query via the Streamlit app interface.

#### Similarity Search:

- The query is converted into an embedding using LLaMA.

- FAISS performs a similarity search to retrieve the most relevant chunks based on vector similarity.

#### Context Preparation:

- Combines the retrieved document chunks with the user’s prompt.

#### Response Generation:

- Sends this combined context to LLaMA via AWS Bedrock.

- Generates a contextual, document-aware response grounded in the retrieved information.


## 🛠️ Tech Stack
**`AWS Bedrock`** – for hosting and interacting with LLaMA

**`LangChain`** – to manage PDF loading, chunking, embeddings, and vector store operations

**`LLaMA`** – for both embedding generation and final response generation

**`FAISS`** – for fast, scalable vector similarity search

**`Streamlit`** – for creating the interactive web interface

**`Python`** – for backend development

