from flask import Flask, request, jsonify
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from datasets import load_dataset
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import requests
import re

# Initialize the Flask application
app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName("Amazon_Reviews_Recommendation").getOrCreate()

# Load the McAuley-Lab/Amazon-Reviews-2023 dataset
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", split="train")

# Convert the dataset to a Spark DataFrame
reviews_df = spark.createDataFrame(dataset)
reviews_df.show(5)

# Load the SentenceTransformer model for product embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract product descriptions (assuming description is available in the dataset) and generate embeddings
product_texts = reviews_df.select(col("reviewText")).rdd.flatMap(lambda x: x).collect()
product_asins = reviews_df.select(col("asin")).rdd.flatMap(lambda x: x).collect()
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


# Load SST-5 dataset (for training the sentiment model)
### Phase 2: Sentiment Analysis with `SetFit/sst5`

# Load SST-5 dataset from SetFit
sst5 = load_dataset("SetFit/sst5", split="train")

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the SST-5 dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_sst5 = sst5.map(tokenize_function, batched=True)
tokenized_sst5.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])  # Use 'text' as input and 'label' as target

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

# Fine-tune the model using SST-5 from SetFit
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

# Apply sentiment analysis to all reviews in the dataset
def analyze_product_reviews(reviews):
    sentiment_scores = [classify_review_sentiment(review) for review in reviews]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)  # Average sentiment score for the product
    return avg_sentiment

# Calculate the average sentiment score for each product (using 'asin' as the product identifier)
product_sentiments = reviews_df.rdd.map(lambda row: (row['asin'], classify_review_sentiment(row['reviewText']))) \
    .groupByKey() \
    .mapValues(lambda sentiments: sum(sentiments) / len(sentiments)) \
    .toDF(["asin", "avg_sentiment"])

# Convert to Pandas for merging with embeddings
product_sentiments = product_sentiments.toPandas()

# Function to enhance product embeddings with sentiment scores
def enhance_embeddings_with_sentiment(product_embeddings, product_sentiments):
    sentiment_scores = np.array(product_sentiments['avg_sentiment']).reshape(-1, 1)
    enhanced_embeddings = np.concatenate([product_embeddings, sentiment_scores], axis=1)
    return enhanced_embeddings

# Enhance product embeddings with sentiment
product_embeddings_with_sentiment = enhance_embeddings_with_sentiment(product_embeddings, product_sentiments)

# Function to perform similarity search using FAISS with sentiment-enhanced embeddings
def get_specific_products_with_sentiment(general_items, num_items):
    # Extract 'name' + 'category' + 'description' for encoding
    item_texts = [f"{item['name']} {item['category']} {item['description']}" for item in general_items]
    
    # Get sentiment-enhanced embeddings
    general_item_embeddings = embedding_model.encode(item_texts)
    sentiment_enhanced_embeddings = enhance_embeddings_with_sentiment(general_item_embeddings, product_sentiments)
    
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
    
    # Step 2: Get specific product recommendations using FAISS and sentiment-enhanced embeddings
    specific_products = get_specific_products_with_sentiment(general_items, 5)
    
    return jsonify({
        "general_recommendations": general_items,
        "specific_recommendations": specific_products
    })

# Start the Flask application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7070)
