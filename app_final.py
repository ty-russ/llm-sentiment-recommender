from flask import Flask, request, jsonify
import faiss
import numpy as np
import mlflow
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from datasets import load_dataset
import json
import requests 
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import ttest_rel

# Initialize the Flask application
app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName("Recommender_System").getOrCreate()

# Load the Amazon Reviews 2023 dataset (User Reviews and Metadata)
user_reviews = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
item_metadata = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty", split="full", trust_remote_code=True)

# Convert datasets to Pandas DataFrames
reviews_df = spark.createDataFrame(user_reviews['full'])
metadata_df = spark.createDataFrame(item_metadata)

# Merge the datasets on the 'asin' column
combined_df = reviews_df.join(metadata_df, "asin", "inner")

# Load the SentenceTransformer model for product embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for product descriptions
product_texts = combined_df.select(col("text")).rdd.flatMap(lambda x: x).collect()
product_embeddings = embedding_model.encode(product_texts)

# Build FAISS index for product embeddings
index = faiss.IndexFlatL2(product_embeddings.shape[1])
index.add(np.array(product_embeddings))  # Add embeddings to the index

# Load GPT-4ALL model for general recommendations
gpt4all_url = "http://localhost:4891/v1/chat/completions"

# Function to generate general product recommendations using GPT-4ALL
def get_general_products_gpt4all(items):
    system_prompt = 'You are an AI assistant functioning as a recommendation system for an e-commerce website. Be specific and limit your answers to the requested format.'
    items_delimited = ', '.join(items[:-1]) + ', and ' + items[-1] if len(items) > 1 else items[0]
    user_prompt = f"A user bought {items_delimited}. What 5 items would they be likely to purchase next? Express response in JSON format with an array of 'next_items'."
    full_prompt = f"{system_prompt}\n{user_prompt}"

    prompt_data = {
        "model": "gpt4all-llama-3-8B-instruct",
        "messages": [{"role": "user", "content": full_prompt}],
        "max_tokens": 2000
    }
    
    try:
        with requests.post(gpt4all_url, json=prompt_data, stream=True) as response:
            response.raise_for_status()
            full_response = ''.join(chunk.decode() for chunk in response.iter_content(chunk_size=8192))
        json_response = json.loads(full_response)
        message_content = json_response.get('choices', [])[0].get('message', {}).get('content', '')
        json_match = re.search(r'```\n(.*?)\n```', message_content, re.DOTALL)
        if json_match:
            parsed_content = json.loads(json_match.group(1))
            return parsed_content.get('next_items', [])
        return []
    except requests.exceptions.RequestException as e:
        print(f"Error making the request: {e}")
        return []

### Phase 2: Sentiment Analysis with SST-5

# Load SST-5 dataset
sst5 = load_dataset("SetFit/sst5", split="train")

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the SST-5 dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_sst5 = sst5.map(tokenize_function, batched=True)
tokenized_sst5.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Load BERT model for sequence classification with 5 sentiment classes
sentiment_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

# Set training arguments for fine-tuning
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
)

# Fine-tune the model using SST-5
trainer = Trainer(
    model=sentiment_model,
    args=training_args,
    train_dataset=tokenized_sst5,
    eval_dataset=tokenized_sst5,
)
trainer.train()

# Function to classify sentiment of a product review
def classify_review_sentiment(review_text):
    inputs = tokenizer(review_text, return_tensors="pt", padding="max_length", truncation=True)
    outputs = sentiment_model(**inputs)
    sentiment_score = torch.argmax(outputs.logits, dim=1).item()  # Returns sentiment label (0-4)
    return sentiment_score

# Function to analyze product reviews for sentiment
def analyze_product_reviews(reviews):
    return [classify_review_sentiment(review) for review in reviews]

# Enhance product embeddings with sentiment analysis
def enhance_embeddings_with_sentiment(product_embeddings, reviews):
    sentiment_scores = [sum(analyze_product_reviews(review)) / len(review) for review in reviews]
    sentiment_array = np.array(sentiment_scores).reshape(-1, 1)
    enhanced_embeddings = np.concatenate([product_embeddings, sentiment_array], axis=1)
    return enhanced_embeddings

# Function to perform similarity search using FAISS with sentiment-enhanced embeddings
def get_specific_products_with_sentiment(general_items, num_items):
    item_texts = [f"{item['name']} {item['category']} {item['description']}" for item in general_items]
    general_item_embeddings = embedding_model.encode(item_texts)
    sentiment_enhanced_embeddings = enhance_embeddings_with_sentiment(general_item_embeddings, product_texts)
    D, I = index.search(np.array(sentiment_enhanced_embeddings), num_items)
    similar_products = [product_texts[i] for i in I[0]]
    return similar_products

# Evaluation Functions

# Function to evaluate the recommendation system using precision, recall, and F1-score
def evaluate_recommendations(true_items, predicted_items):
    true_labels = [1 if item in true_items else 0 for item in predicted_items]
    predicted_labels = [1] * len(predicted_items)  # Assuming all recommended items are relevant
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    return precision, recall, f1

# Evaluate the ROC-AUC for the sentiment model
def evaluate_sentiment_model(test_data):
    y_true = []
    y_pred_prob = []
    for example in test_data:
        inputs = tokenizer(example['text'], return_tensors="pt", padding="max_length", truncation=True)
        outputs = sentiment_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)  # Get probabilities for each class
        y_true.append(example['label'])
        y_pred_prob.append(probs[0][1].item())  # Get probability of the positive class
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    print(f"ROC-AUC Score: {roc_auc}")
    return roc_auc

# Paired t-test for baseline vs enhanced model
def perform_paired_t_test(baseline_metrics, enhanced_metrics):
    t_stat, p_value = ttest_rel(baseline_metrics, enhanced_metrics)
    print(f"t-statistic: {t_stat}, p-value: {p_value}")
    if p_value < 0.05:
        print("Reject H0: The enhanced model statistically outperforms the baseline model.")
    else:
        print("Fail to reject H0: No significant difference between the models.")

# Flask API Endpoints

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    items = data.get("items", [])
    true_items = data.get("true_items", [])  # Provided during evaluation
    
    if not items:
        return jsonify({"error": "No items provided"}), 400
    
    # Step 1: Get general product recommendations using GPT-4ALL
    general_items = get_general_products_gpt4all(items)
    
    # Step 2: Get specific product recommendations using FAISS and sentiment-enhanced embeddings
    specific_products = get_specific_products_with_sentiment(general_items, 5)
    
    # Step 3: Evaluate the recommendations
    precision, recall, f1 = evaluate_recommendations(true_items, specific_products)
    
    return jsonify({
        "general_recommendations": general_items,
        "specific_recommendations": specific_products,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

@app.route('/feedback', methods=['POST'])
def collect_feedback():
    data = request.json
    feedback_scores = data.get("feedback_scores", [])  # List of relevance scores (1-5)
    
    if not feedback_scores or len(feedback_scores) != 5:  # Assuming 5 recommendations
        return jsonify({"error": "Invalid feedback"}), 400
    
    avg_feedback = sum(map(int, feedback_scores)) / len(feedback_scores)
    print(f"Average User Feedback Score: {avg_feedback}/5")
    
    return jsonify({"average_feedback": avg_feedback})

# Start the Flask application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7070)
