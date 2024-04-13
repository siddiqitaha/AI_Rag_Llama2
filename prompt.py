import embedding_RetrievalQA_Wiki
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Build prompt
template = """Use the following pieces of context to answer the question. 
If pieces of context do not contain the answer use your internal knowledge.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum for the answer. Keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Answer:"""


QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain

def get_answer(question, temperature):
    embedding_RetrievalQA_Wiki.set_temperature(temperature)  # Adjust the model's temperature
    
    # Use the `template` variable directly without redeclaration
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)  # Use the existing `template` variable

    qa_chain = RetrievalQA.from_chain_type(
        embedding_RetrievalQA_Wiki.llm,
        retriever=embedding_RetrievalQA_Wiki.db3.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    result = qa_chain.invoke({"query": question})
    return result.get("result", "No response generated.")

# question = "Who is Albert Einstein?"
# result = qa_chain.invoke({"query": question})
# # Check the result of the query
# print(result.get("result", "No response generated."))