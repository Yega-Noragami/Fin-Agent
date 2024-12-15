# Project Overview

The provided code base is a comprehensive Python script that integrates various functionalities, including PDF text extraction, Google search, embedding functions, and conversational agents using the Autogen framework. It is designed to automate and facilitate tasks such as document processing, information retrieval, and interaction with users through conversational agents. Below, I'll break down the key aspects and functionalities of the code base:

### Environment Configuration
The script starts by importing necessary libraries and setting an environment variable to disable the use of Docker for Autogen.

### Autogen and Langchain Imports
It imports modules from the `autogen` library, which is likely a framework for automating tasks or processes. The script also imports from `langchain_community`, which suggests the use of language models and embeddings for processing or generating text.

### PDF Text Extraction
The `extract_text_from_pdf` and `read_contents_of_folder` functions are designed to extract text from PDF files located in a specified folder. This functionality is useful for processing documents and extracting information.

### Google Search
The `GoogleQuery` class encapsulates the functionality to perform Google searches using the Serp API, and extract insights from the search results. This could be used for gathering information from the web based on user queries.

### Conversational Agents
The script defines two main functions, `askStrategyAgent` and `askResearchAgent`, which seem to interact with conversational agents for strategy and research purposes, respectively. These functions utilize embeddings and text splitting for processing queries and generating responses.

### MemGPT Configuration
There is a conditional check for a `MemGPT` variable, which suggests the option to use a MemGPT model, a variant of GPT (Generative Pre-trained Transformer) for memory-augmented conversational agents.

### Main Function
The `main` function orchestrates the overall process, including reading and processing PDF documents, and initializing conversational agents for handling user queries. It demonstrates how to set up and use conversational agents for strategy and research tasks based on user input.

### Autogen Agents and Group Chat
The script sets up Autogen agents and a group chat environment, allowing for the interaction between user proxy agents and assistant agents for strategy and research. This setup facilitates a conversational interface where users can ask questions, and the system retrieves and processes information to generate responses.



# Project Setup Guide

This guide will walk you through setting up a virtual environment using Python 3.11.4 and running the `autogen-rag.py` script.

## Setting Up a Virtual Environment

### Install Python 3.11.4

Ensure that Python 3.11.4 is installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/release/python-3114/) or use a version management tool like `pyenv`.

### Create a Virtual Environment

Open your terminal, navigate to your project directory, and run the following command to create a virtual environment using conda:

~~~bash
conda create -n autogen python=3.11.4 
~~~

### Activate Virtual Environment

To activate the virtual environmentm, use the following command

~~~bash
python3.11 -m venv venv 
~~~


### Install Requirement from File

Once you have activated your virtual environment you have to install the necessary dependencies using the requirements.txt file

~~~bash
 pip install -r requirements.txt
~~~

## Running The Code 

### Run Autogen Chat

The Autogen chat can be acessed after running the file 'autogen-rag.py'.Make sure you have a virtual environment activated. 

~~~bash
 python3 autogen-rag.py
~~~

This Initiates the chat with Strategy and Research Agent which have been initiated in the code. 

### MemGPT

The Memory GPT is currently facing dependency issue with latest autogen Framework and functionality is set to False at the moment , it can be changed in the codebase . 
~~~python
MemGPT = False
~~~


