import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama

# Read the CSV file into a DataFrame
df = pd.read_csv("question_answers_evaluation.csv")

# Assuming 'contexts' is the column containing the document texts
chunks = df['contexts'].tolist()

# Create the FAISS vector store from the document texts
vector_store = FAISS.from_texts(chunks, HuggingFaceEmbeddings())

# Define the function to test local retrieval QA
def test_local_retrieval_qa(model: str):
    chain = RetrievalQA.from_llm(
        llm=ChatOllama(model=model),
        retriever=vector_store.as_retriever(),
    )
    
    predictions = []
    for it, row in tqdm(df.iterrows(), total=len(df)):
        resp = chain.invoke({
            "query": row["question"]
        })
        predictions.append(resp["result"])
    
    df[f"{model}_result"] = predictions

# Test different models
test_local_retrieval_qa("mistral")
test_local_retrieval_qa("llama2")
test_local_retrieval_qa("zephyr")
test_local_retrieval_qa("orca-mini")
test_local_retrieval_qa("phi")

# Save the results to a new CSV file
df.to_csv("./data/cidr_lakehouse_qa_retrieval_prediction.csv", index=False)
