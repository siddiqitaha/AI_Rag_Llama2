# Import required libraries
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Define the prompt template
template = """Use the following pieces of context to answer the question. 
If pieces of context do not contain the answer use your internal knowledge.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum for the answer. Keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Answer:"""

# Build the prompt template
qa_template = PromptTemplate.from_template(template)

# Initialize the RAG-like model
model = "llama2"
llm = ChatOllama(model=model)

# Initialize the embedding retriever
embedder = OllamaEmbeddings(model='nomic-embed-text')

# Set up the retrieval augmented QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=embedder.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_template}
)

# Initialize the base model for comparison
base_llm = ChatOllama(model=model)

# Functions to query both models
def query_rag(question: str) -> str:
    response = qa_chain.ask(question)
    return response.answer if response.answer else "No answer provided."

def query_base_model(question: str) -> str:
    response = base_llm.ask(question)
    return response.answer if response.answer else "No answer provided."

# Sample questions
questions = [
    "What is the capital of France?",
    "Explain the process of photosynthesis.",
    "What is the significance of E=mc^2?"
]

# Query both models and print responses
for question in questions:
    rag_answer = query_rag(question)
    base_answer = query_base_model(question)

    print(f"Question: {question}")
    print(f"RAG Model Answer: {rag_answer}")
    print(f"Base Model Answer: {base_answer}")
    print("\n")
