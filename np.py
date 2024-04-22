import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv("question_answers_evaluation.csv")

# Display the first few rows of the DataFrame to confirm it's loaded correctly
print(df.head())

# Assume your DataFrame has columns 'answer' for the system's responses and 'ground_truth' for the correct answers
predicted_answers = df['answer'].tolist()
ground_truth_answers = df['ground_truth'].tolist()

from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu

# Example function to compute BLEU score for each response against the ground truth
def calculate_bleu_scores(ground_truths, predictions):
    scores = []
    for gt, pred in zip(ground_truths, predictions):
        # Tokenize the sentences if not already, nltk expects lists of tokens
        reference = gt.split()  # Splitting by words
        candidate = pred.split()
        score = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))
        scores.append(score)
    return scores

# Calculate BLEU scores
bleu_scores = calculate_bleu_scores(ground_truth_answers, predicted_answers)
average_bleu_score = sum(bleu_scores) / len(bleu_scores)

# Print average BLEU score
print(f"Average BLEU Score: {average_bleu_score}")

# Assuming a simple match for binary classification
binary_predicted = [1 if pred == gt else 0 for pred, gt in zip(predicted_answers, ground_truth_answers)]
binary_ground_truth = [1] * len(ground_truth_answers)  # Assuming all ground truths are correct responses

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(binary_ground_truth, binary_predicted)
recall = recall_score(binary_ground_truth, binary_predicted)
f1 = f1_score(binary_ground_truth, binary_predicted)

print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

