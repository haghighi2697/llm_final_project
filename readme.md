### LLM-RAG System for Property Insurance Policy
#### Overview
This project aims to develop a Retrieval-Augmented Generation (RAG) system that answers questions related to my home’s property insurance policy. The goal is to create a tool that simplifies understanding complex insurance documents by allowing users to query specific sections interactively.

#### Problem Description
Insurance policies can be long and difficult to navigate, often leading people to overlook important details. This project addresses the challenge by implementing a question-answering system that helps users quickly access relevant information and understand their rights and coverage without having to sift through the entire document.

#### Project Components
PDF File: The original property insurance policy document.
JSON Files: Chunked and embedded versions of the PDF, optimized for efficient retrieval within the RAG system.
Python Files:
01.preprocessing.ipynb: A notebook that processes and transforms the text data from the PDF into a format that is searchable and ready for use in the retrieval system.
src/my_app.py: The main application script that integrates the RAG system and enables user interaction.
#### Workflow
Preprocessing: The data is processed through the 01.preprocessing.ipynb notebook, where the PDF document is transformed into searchable chunks.
Running the Application: The Streamlit-based application can be launched using the command:

Copy code
streamlit run src/my_app.py

This command initializes the user interface, enabling users to submit queries related to the insurance policy.
Deployment: The application is deployed using Streamlit, allowing for interactive querying of the policy document.
#### Functionality
The application provides an intuitive Streamlit interface where users can input queries about specific clauses or regulations in the property insurance policy. The RAG system processes these queries and returns the relevant sections from the document, helping users quickly access the information they need.
