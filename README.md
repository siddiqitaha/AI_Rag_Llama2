# Custom Ollama RAG Model

This project uses Ollama library to run a custom Llama 2 with 7b parameters RAG (Retrieval Augmented Generation) model. The program will launch a custom chatbot which references information and data about The University of Aberdeen from its Wikipedia Page. The project includes a data extraction module, retreival engine module, prompt module, and a gradio module. 

## Overview

The software is designed to query Llama2 model using the Ollama framework:
1. **Base Model (Llama2 7b)** - This model serves as our baseline for generating responses.
2. **Custom RAG Model** - This is the retrieval-augmented generation model which enhances the over all answer quality through additional contextual data.
3. **Data Scraping**

## Features

- Uses the powerful Llama2 7b model.
- Incorporates a custom RAG model.
- Provides a UI to perform queries and test the model.
- Performs a side to side evaluation by running list of questions through both base and RAG models and saves results to a CSV file.


-----
# Collect New Data
Before running the software to collect more data for the Vector Database, ensure you have Python installed along with the necessary dependencies:
- Python 3.8 or newer
- Install [Ollama](https://ollama.com/download/windows)
- Install Git
- Install CUDA (GPU only)

---
### Installation

**First**, Clone Repository to your local machine:
```bash
git clone https://github.com/siddiqitaha/AI_Rag_Llama2.git
cd AI_Rag_Llama2
```
----, Install all neseccerry Libraries and Dependencies 
- Install Python Libraries and Dependencies
```bash
pip install -r requirements.txt
```
**(_Only_ Linux) Third,** run Ollama on a **seperate** terminal
```bash
ollama serve
```

**Finally**, Run Data Extraction and Vector Store:
```bash
python3 chromadb_creation.py
```
Enter a **Wikipedia URL** you would like to collect data from to reference.


----
# Running Chatbot
---
## Windows/Linux (CPU/GPU)
### Prerequisites

Before running the software, ensure you have Python installed along with the necessary dependencies:
- Python 3.8 or newer
- Install [Ollama](https://ollama.com/download/windows)
- Install Git
- Install CUDA (GPU only)
---
### Installation

**First**, Clone Repository to your local machine:
```bash
git clone https://github.com/siddiqitaha/AI_Rag_Llama2.git
cd AI_Rag_Llama2
```
----, Install all neseccerry Libraries and Dependencies 
- Install Python Libraries and Dependencies
```bash
pip install -r requirements.txt
```
**(_Only_ Linux) Third,** run Ollama on a **seperate** terminal
```bash
ollama serve
```

**Finally**, run the application
```bash
python3 gradio_chatbot.py
```
----
Once the Application in running, it will provide you with the following:

#### Example:
- Running on local URL:  http://127.0.0.1:7860
- Running on public URL: https://4e3bcb307590d89f2f.gradio.live


---
