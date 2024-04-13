import llm_check
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import VectorDBQA


llm_check.check_and_pull_nomic_embed_text()
llm_check.check_and_pull_llama2()
sample = "What is Japan's capital city?"
model = "llama2"

# Embedding documents with Ollama using Nomic-embed-text
def llm_embedding(query):
    embedding=OllamaEmbeddings(model='nomic-embed-text')
    db3 = Chroma(persist_directory="./VectorStore", embedding_function=embedding)
    qa=VectorDBQA.from_chain_type(llm=ChatOllama(model=model, temperature=2), k="1", chain_type="refine", vectorstore=db3, )
    response= qa(query, return_only_outputs=True)
    return response

print(llm_embedding(sample))