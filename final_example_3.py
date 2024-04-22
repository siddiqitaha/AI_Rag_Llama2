# Import necessary libraries and modules
import csv
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
import embedding_RetrievalQA_Wiki  # Import your custom retrieval module

# Initialize the RAG model with retrieval
def setup_rag_model():
    # Initialize the language model
    llm = ChatOllama(model="llama2")

    # Initialize the retriever from the custom module
    retriever = embedding_RetrievalQA_Wiki.db3.as_retriever()

    # Set up the retrieval augmented QA chain
    qa_chain = RetrievalQA(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Initialize the baseline model without retrieval
def setup_baseline_model():
    # Initialize the language model
    llm = ChatOllama(model="llama2")
    return llm

# Function to query the models
def query_models(question, rag_model, baseline_model):
    # Retrieve answer from RAG model
    rag_response = rag_model.ask(question)
    rag_answer = rag_response.answer if rag_response.answer else "No answer provided."

    # Retrieve answer from baseline model
    baseline_response = baseline_model.ask(question)
    baseline_answer = baseline_response.answer if baseline_response.answer else "No answer provided."

    return rag_answer, baseline_answer

# Function to save results to CSV
def save_results_to_csv(results, filename="model_comparison_results.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Question", "RAG Model Answer", "Baseline Model Answer"])
        for result in results:
            writer.writerow(result)

# Main function to run the queries
def main():
    # Set up the models
    rag_model = setup_rag_model()
    baseline_model = setup_baseline_model()

    # Example questions to query
    questions = [
        "What is the capital of France?",
        "Explain the process of photosynthesis.",
        "What is the significance of E=mc^2?"
    ]

    results = []

    # Query the models and store responses
    for question in questions:
        rag_answer, baseline_answer = query_models(question, rag_model, baseline_model)
        results.append([question, rag_answer, baseline_answer])
    
    # Save results to CSV
    save_results_to_csv(results)

    print("Results have been saved to 'model_comparison_results.csv'.")

# Execute the main function
if __name__ == "__main__":
    main()
