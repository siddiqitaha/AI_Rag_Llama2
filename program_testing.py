import csv
from langchain_community.llms import Ollama
from prompt import get_answer

llm = Ollama(model="llama2")
all_questions = [
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

results_list = []

# Loop through each question and get answers from both models
for question in all_questions:
    answer_base = llm.invoke(question)
    answer_rag = get_answer(question, "0")  # Assuming get_answer accepts these parameters
    
    results_list.append([question, answer_base, answer_rag])

# Save results to a CSV file
csv_file_path = 'university_of_aberdeen_qa.csv'
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Question', 'Answer from Llama2', 'Answer from get_answer'])
    writer.writerows(results_list)

print(f"Data saved to {csv_file_path}")
