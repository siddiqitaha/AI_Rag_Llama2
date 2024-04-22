import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_bleu_scores(ground_truths, predictions):
    scores = []
    for gt, pred in zip(ground_truths, predictions):
        gt = str(gt).lower().strip() if isinstance(gt, str) else ""
        pred = str(pred).lower().strip() if isinstance(pred, str) else ""
        reference = gt.split()
        candidate = pred.split()
        score = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))
        scores.append(score)
    return scores

def flexible_match(gt, pred):
    gt = gt.lower().strip()
    pred = pred.lower().strip()
    return pred == gt or gt in pred

# Load the CSV file into a DataFrame
df = pd.read_csv("question_answers_evaluation.csv")

# Prepare data for evaluation
predicted_answers = df['answer'].apply(lambda x: str(x) if isinstance(x, str) else "").tolist()
ground_truth_answers = df['ground_truth'].apply(lambda x: str(x) if isinstance(x, str) else "").tolist()

# Calculate BLEU scores
bleu_scores = calculate_bleu_scores(ground_truth_answers, predicted_answers)
average_bleu_score = sum(bleu_scores) / len(bleu_scores)

# Update binary predictions using flexible matching
binary_predicted = [1 if flexible_match(gt, pred) else 0 for gt, pred in zip(ground_truth_answers, predicted_answers)]
binary_ground_truth = [1] * len(ground_truth_answers)  # Assuming all ground truths are correct responses

# Calculate precision, recall, and F1 with zero_division handling
precision = precision_score(binary_ground_truth, binary_predicted, zero_division=0)
recall = recall_score(binary_ground_truth, binary_predicted, zero_division=0)
f1 = f1_score(binary_ground_truth, binary_predicted, zero_division=0)

# Output the results
print(f"Average BLEU Score: {average_bleu_score}")
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
