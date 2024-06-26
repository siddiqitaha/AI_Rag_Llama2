import llm_check
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama

llm_check.check_and_pull_nomic_embed_text()
llm_check.check_and_pull_llama2()
model = "llama2"

def set_temperature(new_temperature):
    global llm
    llm = ChatOllama(model=model, temperature=new_temperature)

# Embedding documents with Ollama using Nomic-embed-text
embedding=OllamaEmbeddings(model='nomic-embed-text')
db3 = Chroma(persist_directory="./VectorStore_PDF", embedding_function=embedding)