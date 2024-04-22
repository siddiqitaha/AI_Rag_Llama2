import pandas as pd
#import chromadb_creation
#import embedding_RetrievalQA_Wiki
from giskard.rag import evaluate


# Data which will be the knowledge base is safed as df to be tested with.

#df = pd.DataFrame([d.page_content for d in chromadb_creation.texts], columns=["text"])
#print(df.head(10))

# Giskard is a library which helps test knowledge bases, it helps with building test cases to test the LLM model.
 
#KnowledgeBase = KnowledgeBase(df)