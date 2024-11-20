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

# Initialize the Flask application
app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName("Recommender_System").getOrCreate()

# Load the Amazon Reviews 2023 dataset (User Reviews and Metadata)
user_reviews = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
item_metadata = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty", split="full", trust_remote_code=True)

# Convert datasets to Pandas DataFrames
# print(user_reviews)
# Convert Hugging Face Dataset to Pandas DataFrame
user_reviews_df = user_reviews['full'].to_pandas()
item_metadata_df = item_metadata.to_pandas()
# print(user_reviews_df)
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
schema = StructType([
    StructField("main_category", StringType(), True),
    StructField("title", StringType(), True),
    StructField("average_rating", StringType(), True),
    StructField("rating_number", IntegerType(), True),
    StructField("features", StringType(), True),
    StructField("description", StringType(), True),
    StructField("price", StringType(), True),
    # StructField("images", StringType(), True),
    StructField("videos", StringType(), True),
    StructField("store", StringType(), True),
    StructField("categories", StringType(), True),
    StructField("details", StringType(), True),
    StructField("parent_asin", StringType(), True),
    StructField("bought_together", StringType(), True),
    StructField("subtitle", StringType(), True),
    StructField("author", StringType(), True),
    StructField("product_text", StringType(), True)
])



# print(item_metadata_df)
# drop columsn
item_metadata_df = item_metadata_df.drop(columns=['images'])
# create product descriptions column
# join column title, main_category, description, categories,features,details 
# Replace NaN values with empty strings and convert each column to a string
item_metadata_df["title"] = item_metadata_df["title"].fillna("").astype(str)
item_metadata_df["main_category"] = item_metadata_df["main_category"].fillna("").astype(str)
item_metadata_df["description"] = item_metadata_df["description"].fillna("").astype(str)

# For columns with lists, join elements with a comma and space
item_metadata_df["categories"] = item_metadata_df["categories"].fillna("").apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
item_metadata_df["features"] = item_metadata_df["features"].fillna("").apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
item_metadata_df["details"] = item_metadata_df["details"].fillna("").apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

# Concatenate the columns to form the product description
item_metadata_df["product_text"] = (
    item_metadata_df["title"] + " " +
    item_metadata_df["main_category"] + " " +
    item_metadata_df["description"] + " " +
    item_metadata_df["categories"] + " " +
    item_metadata_df["features"] + " " +
    item_metadata_df["details"]
)
# print(item_metadata_df)
# Create DataFrame with explicit schema
metadata_df = spark.createDataFrame(item_metadata_df, schema=schema)

user_reviews_df = user_reviews_df.drop(columns=['images'])
reviews_df = spark.createDataFrame(user_reviews_df)
print("Reviews:",list(reviews_df.columns))
print("Product Data",list(metadata_df.columns))






# Merge the datasets on the 'asin' column
combined_df = reviews_df.join(metadata_df, "parent_asin", "inner")

# print(combined_df)
# Load the SentenceTransformer model for product embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
 
# Generate embeddings for product descriptions
product_texts = combined_df.select(col("product_text")).rdd.flatMap(lambda x: x).collect()
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

# Flask route for recommending products
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    items = data.get("items", [])
    
    if not items:
        return jsonify({"error": "No items provided"}), 400
    
    general_items = get_general_products_gpt4all(items)
    specific_products = get_specific_products_with_sentiment(general_items, 5)
    
    return jsonify({
        "general_recommendations": general_items,
        "specific_recommendations": specific_products
})

# Start the Flask application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7070)
