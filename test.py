import prompt
import pandas as pd
from datasets import Dataset
from ragas import evaluate
import embedding_RetrievalQA_Wiki
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
import os


question_list = [
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

ground_truths_list = [
    "1495",
    "William Elphinstone, Bishop of Aberdeen and Chancellor of Scotland",
    "King's College and Marischal College",
    "Initium sapientiae timor domini; The fear of the Lord is the beginning of wisdom",
    "The Crown Tower is an iconic symbol of the university, representing its history and academic excellence",
    "The university contributed to intellectual development, innovation, and the promotion of enlightenment thought",
    "It combines elements from the arms of the founding colleges, reflecting their merger and the university's heritage",
    "It led to a shift from Catholic to Protestant teaching and changes in governance structures",
    "Marischal College played a critical role in promoting enlightenment thought, particularly in philosophy, science, and medicine",
    "It evolved from medieval buildings to modern facilities, expanding significantly in the 20th century",
    "The controversy involves proposed changes to the program due to financial constraints and declining enrollment, with backlash from the academic community",
    "King's College had a religious and educational focus since its Catholic founding, while Marischal College was a Protestant alternative",
    "The university has expanded its curriculum to include a wide range of disciplines beyond its historical focus",
    "This requires analysis of budget, endowment, income from research grants, and expenditures provided in the Wikipedia page or official reports",
    "It underscores the university's significant contributions to global knowledge and research excellence",
    "It involves issues of transparency, accountability, and governance in the use of university funds",
    "The motto suggests a foundational respect for knowledge and wisdom, guiding the university's ethical approach to education and research",
    "Initiatives might include expanding global education programs, interdisciplinary research, and focusing on sustainability",
    "The university should employ strategies to adapt to broader educational demands while emphasizing the importance of language studies",
    "The program would draw on the university's strengths in history, divinity, medicine, and the Scottish Enlightenment"
]

question_context_answers = []

model = "llama2"
llm = ChatOllama(model=model)
# embeddings=OllamaEmbeddings(model='nomic-embed-text'

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt.template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=embedding_RetrievalQA_Wiki.db3.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question_context_answers = []

for question, ground_truth in zip(question_list, ground_truths_list):
    result = qa_chain.invoke({"query": question})
    answer_summary = {
        "question": question,
        "contexts": ["\n".join(doc.page_content for doc in result["source_documents"])],
        "answer": result.get("answer", ""),
        "ground_truths": [ground_truth]  # Adding ground truth here
    }
    question_context_answers.append(answer_summary)

# Convert your list of dictionaries to a pandas DataFrame
df = pd.DataFrame(question_context_answers)

# Make sure the DataFrame includes a 'contexts' column
print(df.columns)  # Should list 'contexts' among others

# Convert your pandas DataFrame to a Hugging Face Dataset
hf_dataset = Dataset.from_pandas(df)

# Now you can pass the Hugging Face dataset to evaluate
results = evaluate(hf_dataset)

# Now, results should hold the evaluation outcome. Depending on what you want to do next,
# you might print it, analyze it, or store it for further processing.
print(results)