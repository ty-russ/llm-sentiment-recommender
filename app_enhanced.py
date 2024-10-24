from flask import Flask, request, jsonify
import faiss
import numpy as np
import mlflow
import mlflow.pyfunc
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import col
import datasets
import json
import requests 
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Initialize the Flask application
app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName("Recommender_System").getOrCreate()

# Load the product dataset from HuggingFace
ds = datasets.load_dataset("xiyuez/red-dot-design-award-product-description")
products_df = spark.createDataFrame(ds['train'])
products_df.show(5)

# Load the SentenceTransformer model for product embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert product text to list and generate embeddings
product_texts = products_df.select(col("text")).rdd.flatMap(lambda x: x).collect()
product_embeddings = embedding_model.encode(product_texts)

# Build FAISS index for product embeddings
index = faiss.IndexFlatL2(product_embeddings.shape[1])  # L2 distance
index.add(np.array(product_embeddings))  # Add embeddings to the index

# Load GPT-4ALL model for general recommendations (running externally)
gpt4all_url = "http://localhost:4891/v1/chat/completions"

# Function to generate general product recommendations using GPT-4ALL
def get_general_products_gpt4all(items):
    system_prompt = 'You are an AI assistant functioning as a recommendation system for an e-commerce website. Be specific and limit your answers to the requested format.'
    
    # Prepare the user prompt, properly formatting the list of items
    items_delimited = ', '.join(items[:-1]) + ', and ' + items[-1] if len(items) > 1 else items[0]
    user_prompt = f"A user bought {items_delimited} in that order. What 5 items would they be likely to purchase next? Express your response as a JSON object with an array of 'next_items', where each item has 'category', 'name', and 'description' fields. Do not include 'id'."
    full_prompt = f"{system_prompt}\n{user_prompt}"

    # The prompt to send to GPT-4ALL
    prompt_data = {
        "model": "gpt4all-llama-3-8B-instruct",
        "messages": [{"role": "user", "content": full_prompt}],
        "max_tokens": 2000  # Increase token limit to allow for a longer response
    }
    
    try:
        with requests.post(gpt4all_url, json=prompt_data, stream=True) as response:
            response.raise_for_status()  # Ensure the request was successful
            
            # Collect the streamed chunks into a single response body
            full_response = ''
            for chunk in response.iter_content(chunk_size=8192):
                full_response += chunk.decode()

        print(f"Raw Response: {full_response}")  # Debugging step

        # Parse the JSON response body
        json_response = json.loads(full_response)

        # Extract the content from the first choice
        message_content = json_response.get('choices', [])[0].get('message', {}).get('content', '')

        # Use regex to extract the JSON part inside the code block
        json_match = re.search(r'```\n(.*?)\n```', message_content, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)  # Extract the actual JSON string

            # Parse the extracted JSON string
            try:
                parsed_content = json.loads(json_str)
                # Extract the 'next_items' list from the parsed content
                next_items = parsed_content.get('next_items', [])
                print(f"Recommended Items: {next_items}")  # Debugging step
                return next_items
            except json.JSONDecodeError:
                print("Failed to parse the content as JSON")
                return []
        else:
            print("No JSON content found in the response")
            return []
    
    except requests.exceptions.RequestException as e:
        print(f"Error making the request: {e}")
        return []


### Phase 2: Sentiment Analysis with SST-5

# Load SST-5 dataset
sst5 = datasets.load_dataset("sst", split="train")

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding="max_length", truncation=True)

# Tokenize the SST-5 dataset
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
    sentiment_score = torch.argmax(outputs.logits, dim=1).item()  # Returns the sentiment label (0-4)
    return sentiment_score

# Apply sentiment analysis to all reviews
def analyze_product_reviews(reviews):
    sentiment_scores = [classify_review_sentiment(review) for review in reviews]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)  # Average sentiment score for the product
    return avg_sentiment

# Function to enhance product embeddings with sentiment scores
def enhance_embeddings_with_sentiment(product_embeddings, reviews):
    # Get average sentiment score for each product
    sentiment_scores = [analyze_product_reviews(product_review_list) for product_review_list in reviews]

    # Append sentiment scores to product embeddings
    sentiment_array = np.array(sentiment_scores).reshape(-1, 1)
    enhanced_embeddings = np.concatenate([product_embeddings, sentiment_array], axis=1)

    return enhanced_embeddings

# Function to perform similarity search using FAISS with sentiment-enhanced embeddings
def get_specific_products_with_sentiment(general_items, num_items):
    # Extract 'name' + 'category' + 'description' for encoding
    item_texts = [f"{item['name']} {item['category']} {item['description']}" for item in general_items]
    
    # Get sentiment-enhanced embeddings
    general_item_embeddings = embedding_model.encode(item_texts)
    sentiment_enhanced_embeddings = enhance_embeddings_with_sentiment(general_item_embeddings, reviews_for_items)
    
    # Perform similarity search using FAISS
    D, I = index.search(np.array(sentiment_enhanced_embeddings), num_items)
    similar_products = [product_texts[i] for i in I[0]]
    
    return similar_products

# Flask route for recommending products
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    items = data.get("items", [])
    
    if not items:
        return jsonify({"error": "No items provided"}), 400
    
    # Step 1: Get general product recommendations using GPT-4ALL
    general_items = get_general_products_gpt4all(items)
    
    # Step 2: Get specific product recommendations using FAISS
    specific_products = get_specific_products_faiss(general_items, 5)
    
    return jsonify({
        "general_recommendations": general_items,
        "specific_recommendations": specific_products
    })


@app.route('/recommend-with-sentiment', methods=['POST'])
def recommend_with_sentiment():
    data = request.json
    items = data.get("items", [])
    
    if not items:
        return jsonify({"error": "No items provided"}), 400
    
    # Step 1: Get general product recommendations using GPT-4ALL
    general_items = get_general_products_gpt4all(items)
    
    # Step 2: Get specific product recommendations using FAISS and sentiment-enhanced embeddings
    specific_products = get_specific_products_with_sentiment(general_items, 5)
    
    return jsonify({
        "general_recommendations": general_items,
        "specific_recommendations": specific_products
    })
# Start the Flask application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7070)

