#final code
from sentence_transformers import SentenceTransformer
# Load the embedding model
model = SentenceTransformer("nli-roberta-large", trust_remote_code=True)
    
# Define a function to generate embeddings
def get_embedding(data):
    """Generates vector embeddings for the given data."""
    embedding = model.encode(data)
    return embedding.tolist()
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
import waitress

# Connect to your Atlas cluster
client = MongoClient("mongodb+srv://projectworkdemo2020:gBoBQogr4VpMa9IB@cluster0.o19y3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&ssl=true")
collection = client["rag_db2"]["test4"]
# Insert documents into the collection
# result = collection.insert_many(docs_to_insert)
from pymongo.operations import SearchIndexModel
import time

# Create your index model, then create the search index
index_name="vector_index1"
search_index_model = SearchIndexModel(
  definition = {
    "fields": [
      {
        "type": "vector",
        "numDimensions": 1024,
        "path": "embedding",
        "similarity": "euclidean"
      }
    ]
  },
  name = index_name,
  type = "vectorSearch"
)

# Wait for initial sync to complete
predicate=None
if predicate is None:
   predicate = lambda index: index.get("queryable") is True

while True:
   indices = list(collection.list_search_indexes(index_name))
   if len(indices) and predicate(indices[0]):
      break
   time.sleep(5)
# print(index_name + " is ready for querying.")# Define a function to run vector search queries
def get_query_results(query,limit=10):
  """Gets results from a vector search query."""

  query_embedding = get_embedding(query)
  
  pipeline = [
      {
            "$vectorSearch": {
              "index": "vector_index1",
              "queryVector": query_embedding,
              "path": "embedding",
              "exact": True,
              "limit": limit
            }
      }, {
            "$project": {
              "_id": 0,
              "text": 1
         }
      }
  ]

  results = collection.aggregate(pipeline)

  array_of_results = []
  for doc in results:
      array_of_results.append(doc)
  return array_of_results

#Final codepart-2##
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
from pymongo import MongoClient
import time
from datetime import datetime
import traceback
from bson import ObjectId
cli = MongoClient("mongodb+srv://projectworkdemo2020:gBoBQogr4VpMa9IB@cluster0.o19y3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&ssl=true")
db = cli['rag_db3']
chat_collection = db['chat_history']
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to the Flask Application!"
# Specify search query, retrieve relevant documents, and convert to string
@app.route('/query', methods=['POST'])
def search_data():
    data = request.json
    user_query = data.get("user_input")
    if not  user_query:
        return jsonify({"error": "No query provided"}), 400
    context_docs = get_query_results(user_query)
    time.sleep(3)
    context_string = " ".join([doc["text"] for doc in context_docs])
    prompt = f"""
    Use the following pieces of context to answer the question at the end.
    { context_string}
    Question: {user_query}
    
"""
    # Authenticate to Hugging Face and access the model
    os.environ["HF_TOKEN"] = "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"
    llm = InferenceClient(
    "mistralai/Mistral-7B-Instruct-v0.3",
    token = os.getenv("HF_TOKEN"))

    # Prompt the LLM (this code varies depending on the model you use)
    output = llm.chat_completion(
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1024
)
    # Store the conversation in MongoDB
    conversation = {
            "user_query": user_query,
            "response": output.choices[0].message.content,
            "liked":False,
            "disliked":False,
            "reason":"null",
            "timestamp": datetime.now()
        }
        
    # Insert the conversation into MongoDB
    result2=chat_collection.insert_one(conversation)
    chat_id = str(result2.inserted_id) 

    # Return the response as a JSON object
    return jsonify({"response": output.choices[0].message.content,"chat_id":chat_id})
@app.route('/update-chat', methods=['POST'])
def update_chat_history():
    data = request.json
    chat_id = data.get('chat_id')  # Retrieve chat_id
    print(data)  # This will help you debug

    if chat_id:
        try:
            # Convert the chat_id to ObjectId
            object_id = ObjectId(chat_id)
        except Exception as e:
            return jsonify({'message': 'Invalid Chat ID format'}), 400

        result = chat_collection.update_one(
            {'_id': object_id},  # Use _id for MongoDB
            {
                '$set': {
                    'liked': data.get('liked', False),
                    'disliked': data.get('disliked', False),
                    'reason': data.get('reason', None),
                }
            }
        )

        if result.modified_count > 0:
            return jsonify({'message': 'Feedback updated successfully'})
        else:
            return jsonify({'message': 'No changes made or chat not found'}), 404
    else:
        return jsonify({'message': 'Chat ID is required'}), 400
    # Run the Flask app
if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
