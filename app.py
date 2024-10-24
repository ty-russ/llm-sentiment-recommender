
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
# Initialize the Flask application
app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName("Recommender_System").getOrCreate()

# Load the product dataset from HuggingFace
ds = datasets.load_dataset("xiyuez/red-dot-design-award-product-description")
products_df = spark.createDataFrame(ds['train'])
products_df.show(5)

# Load the SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert product text to list and generate embeddings
product_texts = products_df.select(col("text")).rdd.flatMap(lambda x: x).collect()
product_embeddings = embedding_model.encode(product_texts)

print(product_embeddings)

# Build FAISS index
index = faiss.IndexFlatL2(product_embeddings.shape[1])  # L2 distance
index.add(np.array(product_embeddings))  # Add embeddings to the index

# Log model to MLflow
with mlflow.start_run() as run:
    mlflow.sentence_transformers.log_model(
        embedding_model,
        artifact_path="model",
        input_example=product_texts[:5]
    )
    run_id = run.info.run_id
    print(f"Model logged in MLflow run: {run_id}")

# Load the logged MLflow model
model_uri = f"runs:/{run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)

# Load GPT-4ALL model for general recommendations
#gpt4all_model = GPT4All("model/ggml-gpt4all-l13b-snoozy.bin")

# URL of GPT-4ALL running on your local machine
#gpt4all_url = "http://host.docker.internal:4891/v1/chat/completions"  # Update the port if needed
gpt4all_url = "http://localhost:4891/v1/chat/completions"
# Function to generate general product recommendations using GPT-4ALL
# # def get_general_products_gpt4all(items):
#     system_prompt = 'You are an AI assistant functioning as a recommendation system for an e-commerce website. Be specific and limit your answers to the requested format.'
#     items_delimited = ', '.join(items[:-1]) + ', and ' + items[-1] if len(items) > 1 else items[0]
#     user_prompt = f"A user bought {items_delimited} in that order. What five items would they be likely to purchase next? Express your response as a JSON object with a key of 'next_items' and a value representing your array of recommended items."
    
#     prompty = f"{system_prompt}\n{user_prompt}"

#     # Generate a response using GPT-4ALL
#     #response = gpt4all_model.generate(prompt)
    
   
    
#     prompt = {
#     #   "model": "gpt4all-lora-quantized",
#       "model": "gpt4all-llama-3-8B-instruct",
#       "messages": [{"role": "user", "content": prompty}]
#     }
#     with requests.post(gpt4all_url, json=prompt, stream=True) as response:
#         response.raise_for_status()
#         ret = ''
#         for chunk in response.iter_content(chunk_size=8192):
#             ret += chunk.decode()
#         print(ret)
#     # Send the prompt to GPT-4ALL running on your host machine via HTTP request
#     #response = requests.post(gpt4all_url, json=prompt,timeout=120)
#     print(response)
#     if response.status_code != 200:
#         return []
     
#     # Extract the list from the response (assuming response is in JSON format)
#     json_response = response.json()
#     print(json_response)
#     # ret = json_response.get('next_items', [])
#     # Navigate to the 'content' part of the response
#     message_content = json_response.get('choices', [])[0].get('message', {}).get('content', '')

#     # The content is a stringified JSON, so we need to parse it
#     try:
#         parsed_content = json.loads(message_content)
#         # Extract the 'next_items' from the parsed content
#         ret = parsed_content.get('next_items', [])
#         print(ret)
#         return(ret)
#     except json.JSONDecodeError:
#         print("Failed to parse the content as JSON")
        
        
#         # Extract the list from the response (assuming response is in JSON format)
#         # json_text = response[:response.rindex('}')+1]  # Extract the JSON part
#         # ret = json.loads(json_text)['next_items']


# def get_general_products_gpt4all(items):
#     system_prompt = 'You are an AI assistant functioning as a recommendation system for an e-commerce website. Be specific and limit your answers to the requested format.'
    
#     # Prepare the user prompt, properly formatting the list of items
#     items_delimited = ', '.join(items[:-1]) + ', and ' + items[-1] if len(items) > 1 else items[0]
#     user_prompt = f"A user bought {items_delimited} in that order. What five items would they be likely to purchase next? Express your response as a JSON object with a key of 'next_items' and a value representing your array of recommended items."
    
#     full_prompt = f"{system_prompt}\n{user_prompt}"

#     # The prompt to send to GPT-4ALL
#     prompt_data = {
#         "model": "gpt4all-llama-3-8B-instruct",
#         "messages": [{"role": "user", "content": full_prompt}],
#         "max_tokens": 2000
#     }
    
#     try:
#         with requests.post(gpt4all_url, json=prompt_data, stream=True) as response:
#             response.raise_for_status()  # Ensure the request was successful
            
#             # Collect the streamed chunks into a single response body
#             full_response = ''
#             for chunk in response.iter_content(chunk_size=8192):
#                 full_response += chunk.decode()

#         print(f"Raw Response: {full_response}")  # Debugging step

#         # Parse the JSON response body
#         json_response = json.loads(full_response)
#         print(f"JSon resp: {json_response}")  # Debugging step

#         # Extract the content from the first choice
#         message_content = json_response.get('choices', [])[0].get('message', {}).get('content', '')
#         print(f"message Items: {message_content}")  # Debugging step

#         # The content is expected to be a stringified JSON, so parse it
#         try:
#             parsed_content = json.loads(message_content)
#             # Extract the 'next_items' list from the parsed content
#             next_items = parsed_content.get('next_items', [])
#             print(f"Recommended Items: {next_items}")  # Debugging step
#             return next_items
#         except json.JSONDecodeError:
#             print("Failed to parse the content as JSON")
#             return []
    
#     except requests.exceptions.RequestException as e:
#         print(f"Error making the request: {e}")
#         return []




def get_general_products_gpt4all(items):
    system_prompt = 'You are an AI assistant functioning as a recommendation system for an e-commerce website. Be specific and limit your answers to the requested format.'
    
    # Prepare the user prompt, properly formatting the list of items
    items_delimited = ', '.join(items[:-1]) + ', and ' + items[-1] if len(items) > 1 else items[0]
    #user_prompt = f"A user bought {items_delimited} in that order. What ten items would they be likely to purchase next? Express your response as a JSON object with a key of 'next_items' and a value representing your array of recommended items."
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




# Function to perform similarity search using FAISS
def get_specific_products_faiss(general_items, num_items):
    # Extract only the 'name' or 'name' + 'category' fields for encoding
    item_texts = [f"{item['name']} {item['category']} {item['description']}" for item in general_items]
    general_item_embeddings = embedding_model.encode(item_texts)
    D, I = index.search(np.array(general_item_embeddings), num_items)
    similar_products = [product_texts[i] for i in I[0]]
    return similar_products


# Load and Preprocess the SST-5 Dataset
#The SST-5 dataset contains 5 sentiment classes: negative, somewhat negative, neutral, somewhat positive, and positive. We will train a BERT-based model to classify product reviews into these sentiment classes.
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load SST-5 dataset
sst5 = load_dataset("sst", split="train")

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding="max_length", truncation=True)

# Tokenize the SST-5 dataset
tokenized_sst5 = sst5.map(tokenize_function, batched=True)
tokenized_sst5.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Load BERT model for sequence classification with 5 sentiment classes
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)


# Fine-Tune the Sentiment Model
# fine-tune the BERT-based sentiment model using SST-5. The model will learn to classify review texts into the 5 sentiment categories.

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_sst5,
    eval_dataset=tokenized_sst5,
)

# Fine-tune the model
trainer.train()


# Apply the Sentiment Model to Product Reviews
# apply the trained sentiment model to classify the customer reviews of the products in your dataset.


import torch

# Define a function to classify sentiment of a product review
def classify_review_sentiment(review_text):
    inputs = tokenizer(review_text, return_tensors="pt", padding="max_length", truncation=True)
    outputs = model(**inputs)
    sentiment_score = torch.argmax(outputs.logits, dim=1).item()  # Returns the sentiment label (0-4)
    return sentiment_score

# Apply sentiment analysis to all reviews
def analyze_product_reviews(reviews):
    sentiment_scores = [classify_review_sentiment(review) for review in reviews]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)  # Average sentiment score for the product
    return avg_sentiment

# Enhance Product Embeddings with Sentiment Scores
# Now that we have sentiment scores for each product (based on customer reviews), 
# we can enhance the product embeddings by incorporating the sentiment score. 
# One approach is to add the sentiment score as an additional dimension in the product embeddings.

def enhance_embeddings_with_sentiment(product_embeddings, reviews):
    # Get average sentiment score for each product
    sentiment_scores = [analyze_product_reviews(product_review_list) for product_review_list in reviews]

    # Append sentiment scores to product embeddings
    sentiment_array = np.array(sentiment_scores).reshape(-1, 1)
    enhanced_embeddings = np.concatenate([product_embeddings, sentiment_array], axis=1)

    return enhanced_embeddings


## Modify FAISS Search to Use Sentiment-Enhanced Embeddings
## Once we have enhanced embeddings that include sentiment information, 
# modify the FAISS similarity search to prioritize items with more positive sentiment scores


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




#--------------------------------------------------
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



# Flask route for recommending products
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
