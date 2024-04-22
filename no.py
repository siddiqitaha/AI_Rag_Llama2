import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_bleu_scores(ground_truths, predictions):
    scores = []
    for gt, pred in zip(ground_truths, predictions):
        # Ensure both ground truth and prediction are strings
        gt = str(gt) if isinstance(gt, str) else ""
        pred = str(pred) if isinstance(pred, str) else ""

        # Tokenize the sentences, nltk expects lists of tokens
        reference = gt.split()  # Splitting by words
        candidate = pred.split()
        # Calculate BLEU score, using equal weights for simplicity
        score = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))
        scores.append(score)
    return scores

def calculate_precision_recall_f1(binary_ground_truth, binary_predicted):
    precision = precision_score(binary_ground_truth, binary_predicted)
    recall = recall_score(binary_ground_truth, binary_predicted)
    f1 = f1_score(binary_ground_truth, binary_predicted)
    return precision, recall, f1

# Load the CSV file into a DataFrame
df = pd.read_csv("question_answers_evaluation.csv")

# Prepare data for evaluation
predicted_answers = df['answer'].apply(lambda x: str(x) if isinstance(x, str) else "").tolist()
ground_truth_answers = df['ground_truth'].apply(lambda x: str(x) if isinstance(x, str) else "").tolist()

# Calculate BLEU scores
bleu_scores = calculate_bleu_scores(ground_truth_answers, predicted_answers)
average_bleu_score = sum(bleu_scores) / len(bleu_scores)

# Assuming binary classification for precision, recall, F1 calculations (this part might need adjustments)
binary_predicted = [1 if pred == gt else 0 for pred, gt in zip(predicted_answers, ground_truth_answers)]
binary_ground_truth = [1] * len(ground_truth_answers)  # Assuming all ground truths are correct responses

# Calculate precision, recall, and F1 score
precision, recall, f1 = calculate_precision_recall_f1(binary_ground_truth, binary_predicted)

# Output the results
print(f"Average BLEU Score: {average_bleu_score}")
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
