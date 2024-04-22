import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import precision_score, recall_score, f1_score
import re
from fuzzywuzzy import fuzz

def normalize_text(text):
    return re.sub(r'\W+', '', text.lower()).strip()

def flexible_match(gt, pred, threshold):
    gt = normalize_text(gt)
    pred = normalize_text(pred)
    return pred == gt or gt in pred or fuzz.token_set_ratio(gt, pred) > threshold

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

# Load the CSV file into a DataFrame
df = pd.read_csv("question_answers_evaluation.csv")

# Prepare data for evaluation
predicted_answers = df['answer'].apply(lambda x: str(x) if isinstance(x, str) and x.strip() else "").tolist()
ground_truth_answers = df['ground_truth'].apply(lambda x: str(x) if isinstance(x, str) else "").tolist()

# Calculate BLEU scores and metrics
bleu_scores = calculate_bleu_scores(ground_truth_answers, predicted_answers)
average_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

# Experiment with different similarity thresholds
thresholds = [60, 70, 80, 90]  # Adjust as needed
best_precision = 0
best_recall = 0
best_f1 = 0
best_threshold = 0

for threshold in thresholds:
    binary_predicted = [1 if flexible_match(gt, pred, threshold) else 0 for gt, pred in zip(ground_truth_answers, predicted_answers)]
    binary_ground_truth = [1] * len(ground_truth_answers)

    # Calculate precision, recall, and F1 scores
    precision = precision_score(binary_ground_truth, binary_predicted, zero_division=1)
    recall = recall_score(binary_ground_truth, binary_predicted, zero_division=1)
    f1 = f1_score(binary_ground_truth, binary_predicted, zero_division=1)

    # Update best scores and threshold if needed
    if f1 > best_f1:
        best_precision = precision
        best_recall = recall
        best_f1 = f1
        best_threshold = threshold

# Output the results
print(f"Average BLEU Score: {average_bleu_score}")
print(f"Best Precision: {best_precision}, Best Recall: {best_recall}, Best F1 Score: {best_f1}, Best Threshold: {best_threshold}")
