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
    "When was the University of Aberdeen established?",
    "Who founded King's College, and what was their role in Scotland?",
    "Name two colleges that merged to form the modern University of Aberdeen.",
    "What motto does the University of Aberdeen adopt, and what is its English translation?",
    "Describe the significance of the Crown Tower at King's College campus.",
    "Based on its founding principles, how has the University of Aberdeen contributed to the Scottish Enlightenment?",
    "How does the university's coat of arms reflect its history and founding colleges?",
    "Discuss the impact of the Scottish Reformation on King's College.",
    "Explain the role of Marischal College during the Scottish Enlightenment.",
    "Outline the development of the University of Aberdeen's campus and buildings from its establishment to the 21st century.",
    "Provide a detailed account of the modern languages controversy at the University of Aberdeen, including the stakeholders' responses.",
    "Compare the roles of King's College and Marischal College in the university's history.",
    "How does the University of Aberdeen's approach to modern language teaching compare to its historical emphasis on medicine and divinity?",
    "Based on the financial data provided, evaluate the university's financial health and research funding in 2021–22.",
    "Interpret the significance of the University of Aberdeen having five Nobel laureates associated with it.",
    "Discuss the ethical considerations surrounding the university's handling of the financial settlement agreement with the former Principal Sir Ian Diamond.",
    "How does the university's motto, 'Initium sapientiae timor domini,' influence its approach to education and research?",
    "Based on current trends and historical developments, what future initiatives might the University of Aberdeen undertake to maintain its standing and contribute to global education?",
    "Considering the controversy over the modern languages department, what strategies should the university employ to address the challenges of declining interest in traditional language studies?",
    "Imagine you are tasked with designing a new interdisciplinary program for the University of Aberdeen that incorporates its rich history. What would be the focus of this program, and how would it draw from the university's strengths?"
]

# Query both models and print responses
for question in questions:
    rag_answer = query_rag(question)
    base_answer = query_base_model(question)

    print(f"Question: {question}")
    print(f"RAG Model Answer: {rag_answer}")
    print(f"Base Model Answer: {base_answer}")
    print("\n")
