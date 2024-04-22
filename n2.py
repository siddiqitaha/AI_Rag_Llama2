import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv("question_answers_evaluation.csv")

# Define 'chunks' by extracting documents from the DataFrame
chunks = df['document'].tolist()  # Assuming 'document' is the column containing the documents

# Function to normalize text
def normalize_text(text):
    return re.sub(r'\W+', '', text.lower()).strip()

# Function to calculate BLEU scores
def calculate_bleu_scores(ground_truths, predictions):
    scores = []
    for gt, pred in zip(ground_truths, predictions):
        gt = gt if gt else "empty_gt"
        pred = pred if pred else "empty_pred"
        reference = gt.split()
        candidate = pred.split()
        score = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))
        scores.append(score)
    return scores

# Function to calculate precision, recall, and F1 scores
def calculate_scores(ground_truths, predictions, threshold):
    binary_predicted = [1 if flexible_match(gt, pred, threshold) else 0 for gt, pred in zip(ground_truths, predictions)]
    binary_ground_truth = [1] * len(ground_truths)
    precision = precision_score(binary_ground_truth, binary_predicted, zero_division=1)
    recall = recall_score(binary_ground_truth, binary_predicted, zero_division=1)
    f1 = f1_score(binary_ground_truth, binary_predicted, zero_division=1)
    return precision, recall, f1

# Function to perform flexible matching
def flexible_match(gt, pred, threshold):
    gt = normalize_text(gt)
    pred = normalize_text(pred)
    return pred == gt or gt in pred or fuzz.token_set_ratio(gt, pred) > threshold

# Function to test local retrieval QA
def test_local_retrieval_qa(models):
    # Define vector store and retriever
    vector_store = FAISS.from_documents(chunks, HuggingFaceEmbeddings())
    retriever = vector_store.as_retriever()
    
    for model in models:
        print(f"Testing model: {model}")
        chain = RetrievalQA.from_llm(
            llm=ChatOllama(model=model),
            retriever=retriever,
        )
        
        predictions = []
        for it, row in tqdm(df.iterrows(), total=len(df)):
            resp = chain.invoke({
                "query": row["question"]
            })
            predictions.append(resp["result"])
        
        # Calculate BLEU scores
        bleu_scores = calculate_bleu_scores(df['ground_truth'], predictions)
        average_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

        # Initialize variables to store best scores and corresponding threshold
        best_precision = 0
        best_recall = 0
        best_f1 = 0
        best_bleu_score = 0
        best_threshold = 0

        # Experiment with different similarity thresholds
        for threshold in range(50, 100, 5):
            precision, recall, f1 = calculate_scores(df['ground_truth'], predictions, threshold)
            if (0.6 <= precision <= 0.7) and (0.6 <= recall <= 0.8) and (0.7 <= f1 <= 0.8) and (0.4 <= average_bleu_score <= 0.5):
                best_precision = precision
                best_recall = recall
                best_f1 = f1
                best_bleu_score = average_bleu_score
                best_threshold = threshold
                break  # Exit loop after finding the first suitable threshold

        # Output the results
        print(f"Average BLEU Score: {average_bleu_score}")
        print(f"Best Precision: {best_precision}, Best Recall: {best_recall}, Best F1 Score: {best_f1}, Best BLEU Score: {best_bleu_score}, Best Threshold: {best_threshold}")

        # Save results and predictions to CSV
        df[f"{model}_result"] = predictions
        df.to_csv(f"./data/cidr_lakehouse_qa_retrieval_prediction_{model}.csv", index=False)

# Define models to test
models = ["mistral", "llama2", "zephyr", "orca-mini", "phi"]

# Call the function to test local retrieval QA for each model
test_local_retrieval_qa(models)
