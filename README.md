# Ollama RAG Model Comparison

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

# Running Chatbot
## Windows/Linux (CPU)
### Prerequisites

Before running the software, ensure you have Python installed along with the necessary dependencies:
- Python 3.8 or newer
- Install [Ollama](https://ollama.com/download/windows)

### Installation

First, clone the repository to your local machine:

- Install Git and Clone Repository
```bash
git clone https://github.com/siddiqitaha/AI_Rag_Llama2.git
cd AI_Rag_Llama2
```
- Install Python Libraries and Dependencies
```bash
pip install -r requirements.txt
```
- Install Python Libraries and Dependencies
```bash
python3 gradio_chatbot.py
```


---

---
