# Ollama RAG Model Comparison

This project uses Ollama library to run a custom Llama 2 with 7b parameters RAG (Retrieval Augmented Generation) model. The program will launch a custom chatbot which references information and data about The University of Aberdeen from its Wikipedia Page. The project includes a data extraction module, retreival engine module, prompt module, and a gradio module. 

## Overview

The software is designed to query Llama2 model using the Ollama framework:
1. **Base Model (Llama2 7b)** - This model serves as our baseline for generating responses.
2. **Custom RAG Model** - This is the retrieval-augmented generation model which enhances the over all answer quality through additional contextual data.

The system processes a predefined list of questions, retrieves answers from both models, and saves the results into a CSV file for easy comparison and analysis.

## Features

- Uses the powerful Llama2 7b model.
- Incorporates a custom RAG model for enhanced retrieval capabilities.
- Compares answers from both models side-by-side.
- Exports the comparison results to a CSV file.

## Prerequisites

Before running the software, ensure you have Python installed along with the necessary dependencies:
- Python 3.8 or newer
- pip (Python package installer)

## Installation

First, clone the repository to your local machine:

```bash
git clone https://your-repository-url.git
cd your-project-directory

---
title: gradio_chatbot.py
app_file: gradio_chatbot.py
sdk: gradio
sdk_version: 4.19.2
---

Install Mini-Conda
Create Local Environment
Python 3.11 required for testing
pip install "giskard[llm]" -U

