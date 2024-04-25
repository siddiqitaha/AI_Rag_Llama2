# import PyPDF2
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

persist_directory = './VectorStore'

def loader():
    loader = WebBaseLoader([input('Wikipedia Link: '), 
                            "https://en.wikipedia.org/wiki/University_of_Aberdeen",
                            "https://www.topuniversities.com/universities/university-aberdeen",
                            "https://www.scotland.org/study/scottish-universities/university-of-aberdeen",
                            "https://www.shiksha.com/studyabroad/uk/universities/university-of-aberdeen"])
    document = loader.load()
    return document

content = loader()[0].page_content

text_splitter = CharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=75,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([content])

# Embedding split documents with Ollama using Nomic-embed-text
embedding=OllamaEmbeddings(model='nomic-embed-text')
vectorstore = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

vectorstore.persist() # Update Vector DB Locally
vectorstore= None # Delete previous Vector DB Files
