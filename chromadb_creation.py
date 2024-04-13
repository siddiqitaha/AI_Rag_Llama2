# import PyPDF2
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

persist_directory = './VectorStore'

def loader():
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/University_of_Aberdeen", )
    document = loader.load()
    return document

content = loader()[0].page_content

text_splitter = CharacterTextSplitter(
    separator="\n", 
    chunk_size=500,
    chunk_overlap=75,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([content])

# Embedding split documents with Ollama using Nomic-embed-text
embedding=OllamaEmbeddings(model='nomic-embed-text')
vectorstore = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

vectorstore.persist() # Update Vector DB Locally
vectorstore= None # Delete previous Vector DB Files