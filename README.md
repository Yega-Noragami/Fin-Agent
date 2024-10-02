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


