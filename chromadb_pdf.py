import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

persist_directory = './VectorStore_PDF'

def load_pdf_content(pdf_path):
    """
    Extract text from a PDF file.
    """
    text = ''
    with open(pdf_path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

pdf_path = "./Data/University_of_Aberdeen.pdf"
content = load_pdf_content(pdf_path)

text_splitter = CharacterTextSplitter(
    separator="\n", 
    chunk_size=500,
    chunk_overlap=75,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([content])

# Embedding split documents with Ollama using Nomic-embed-text
embedding = OllamaEmbeddings(model='nomic-embed-text')
vectorstore = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

vectorstore.persist()  # Update Vector DB Locally
vectorstore = None  # Delete previous Vector DB Files