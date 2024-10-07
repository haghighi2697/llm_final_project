import streamlit as st
import json
from elasticsearch import Elasticsearch
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import ElasticsearchRetriever
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF for PDF processing
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Initialize Elasticsearch connection
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
index_name = 'insurance_chunks'

# Ensure the Elasticsearch index exists, otherwise, create it
if not es.indices.exists(index=index_name):
    st.error("Elasticsearch index not found! Please upload a PDF document to index.")
else:
    st.success(f"Elasticsearch index '{index_name}' found.")

# Function to index the PDF file into Elasticsearch
def index_pdf_to_elasticsearch(pdf_file, index_name):
    doc = fitz.open(pdf_file)
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        # Index each page content into Elasticsearch
        es.index(index=index_name, body={"page_num": page_num, "text": text})
    st.success(f"PDF '{pdf_file.name}' indexed successfully into Elasticsearch.")

# Streamlit interface
st.title("Insurance Query System")
st.write("Welcome! You can upload a PDF and ask questions related to Property Insurance.")

# Step 1: PDF Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    pdf_file_path = f"uploaded_{uploaded_file.name}"
    with open(pdf_file_path, "wb") as f:
        f.write(uploaded_file.read())
    index_pdf_to_elasticsearch(pdf_file_path, index_name)

# Step 2: Input box for user query
query = st.text_input("Enter your question related to the Property Insurance:", "")

# Button to trigger the query
if query:
    embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Setup Elasticsearch retriever
    retriever = ElasticsearchRetriever(
        es=es,
        index_name=index_name,
        embedding_model=embedding_model,
        k=5  # Number of documents to retrieve
    )
    
    # Initialize the LLM (GPT)
    llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # You can change to map_reduce, etc.
        retriever=retriever
    )

    # If the query is entered, process it and get the answer
    if query:
        with st.spinner("Retrieving answer..."):
            answer = qa_chain.run(query)
            st.success(f"Answer: {answer}")

st.sidebar.title("About")
st.sidebar.info("This tool helps you query a property insurance PDF document.")
