import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import precision_score, recall_score, f1_score
import re

def normalize_text(text):
    """Normalize text by lowering case, removing non-word characters and stripping spaces."""
    return re.sub(r'\W+', '', text.lower()).strip()

def flexible_match(gt, pred):
    """Compare normalized predicted text to normalized ground truth, checking for exact matches."""
    gt = normalize_text(gt)
    pred = normalize_text(pred)
    return pred == gt

def calculate_bleu_scores(ground_truths, predictions):
    """Calculate BLEU scores for each pair of ground truth and prediction."""
    scores = []
    for gt, pred in zip(ground_truths, predictions):
        if gt and pred:  # Ensure neither is empty
            reference = gt.split()
            candidate = pred.split()
            score = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))
            scores.append(score)
        else:
            scores.append(0)  # Append zero if any string is empty
    return scores

# Load the CSV file into a DataFrame
df = pd.read_csv("question_answers_evaluation.csv")

# Prepare data for evaluation
predicted_answers = df['answer'].apply(lambda x: str(x) if isinstance(x, str) else "").tolist()
ground_truth_answers = df['ground_truth'].apply(lambda x: str(x) if isinstance(x, str) else "").tolist()

# Print sample data to review
for i, (pred, gt) in enumerate(zip(predicted_answers, ground_truth_answers)):
    if i < 10:  # Print first 10 pairs for inspection
        print(f"Predicted: {pred} - Ground Truth: {gt}")

# Calculate BLEU scores
bleu_scores = calculate_bleu_scores(ground_truth_answers, predicted_answers)
average_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

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
