import retrieval_wiki
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama

model = "llama2"

def set_temperature(new_temperature):
    global llm
    llm = ChatOllama(model=model, temperature=new_temperature)

# Build prompt
template = """Use the following pieces of context to answer the question. 
If pieces of context do not contain the answer use your internal knowledge.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum for the answer. Keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Answer:"""


prompt_chain= PromptTemplate.from_template(template)# Run chain

def get_answer(question, temperature):
    set_temperature(temperature)  # Adjust the model's temperature
    prompt_chain = PromptTemplate.from_template(template)  # Use template variable
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retrieval_wiki.db3.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_chain})
    result = qa_chain.invoke({"query": question})
    return result.get("result", "No response generated.")
