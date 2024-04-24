# Ollama RAG Model Comparison

This project utilizes the Ollama library to compare responses from two different models: a base model using Llama2 with 7b parameters and a custom RAG (Retrieval-Augmented Generation) model. The goal is to evaluate and compare the outputs from both models on a series of questions related to the University of Aberdeen.

## Overview

The software is designed to query two models using the Ollama framework:
1. **Base Model (Llama2 7b)** - This model serves as our baseline for generating responses.
2. **Custom RAG Model** - This is a retrieval-augmented generation model that aims to enhance answer quality by leveraging additional contextual data.

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

