import json
import os
import requests
from django.http import JsonResponse
from sentence_transformers import SentenceTransformer, util
import torch
from .rag_utils import run_rag,run_rag_history
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# print("GOOGLE_API_KEY =", GOOGLE_API_KEY)  # For debugging only!
DATA_DIR = 'tripplanner_backend/data'
HISTORY_FILE = os.path.join(DATA_DIR,'history.json')  
THRESHOLD = 0.7

def get_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except json.JSONDecodeError:
        print("[RAG HISTORY ERROR] Failed to parse JSON history file.")
        return []

def find_similar_query(query,history):
    """
    Find similar queries in the history using Sentence Transformers.
    """
    # Load the model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Load history embeddings
    history_embeddings = []
    for item in history:
        if 'query' in item:
            history_embeddings.append(item['query'])

    # Encode the history queries
    history_embeddings = model.encode(history, convert_to_tensor=True)

    # Calculate cosine similarity
    cosine_scores = util.pytorch_cos_sim(query_embedding, history_embeddings)[0]

    # Find similar queries
    similar_queries = []
    for i, score in enumerate(cosine_scores):
            similar_queries.append((history[i], score.item()))
    if len(similar_queries) > 0:
        # Sort by similarity score
        similar_queries = sorted(similar_queries, key=lambda x: x[1], reverse=True)
        history = [item for item, _ in similar_queries]
        updated_history(history)
    return similar_queries

def clean_filename(query):
    """
    Clean the filename by replacing spaces with underscores and removing special characters.
    """
    # Replace spaces with underscores
    query = query.replace(" ", "_")
    # Remove special characters
    query = ''.join(e for e in query if e.isalnum() or e == '_')
    return f"{query}.json"
def updated_history(history):
    """
    Update the history file with the new history.
    """
    # Save updated history
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def rag_answer_past(query,history):
    try:
        answer =run_rag_history(query,history)
        return JsonResponse(answer, safe=False)
    except Exception as e:
        print(f"Error in rag_answer_past: {e}")
        return JsonResponse({"error": "Failed to get answer from RAG"}, status=500)
def rag_answer(query):
    try:
        answer =run_rag(query)
        return JsonResponse(answer, safe=False)
    except Exception as e:
        print(f"Error in rag_answer: {e}")
        return JsonResponse({"error": "Failed to get answer from RAG"}, status=500)
def update_history(query,filename):
    """
    Update the history file with the new filename.
    """
    # Load existing history
    history = get_history()
    # Add new filename to history
    # history.append(filename)
    history.append({"query": query, "filename": filename})

    # Save updated history
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# Create your views here.
def search(request):
    if request.method == 'GET':
        # Extract query parameters
        query = request.GET.get('query', '')
        if not query:
            return JsonResponse({"error": "Missing query parameter"}, status=400)
        history=get_history()
        if len(history) > 0:
            similar_queries=find_similar_query(query,history)
            if len(similar_queries) > 0 and similar_queries[0][1] > THRESHOLD:
                return rag_answer_past(query,similar_queries[0][0])
                # return JsonResponse({"message": "Using similar query from history"}, status=200)  
    
        # Passing the query to the Google Maps API to Get results
        # url=f"https://maps.googleapis.com/v1/place:searchText"
        # headers = {
        # 'Content-Type': 'application/json',
        # "X-Goog-Api-Key": GOOGLE_API_KEY,
        # # "X-Goog-FieldMask": "*"
        # "X-Goog-FieldMask": "name,formatted_address,geometry/location,description,photos"
        # }
        url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
        'Content-Type': 'application/json',
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.photos,places.editorialSummary,places.types,places.rating"#replaced the shortDescription with editorialSummary
        }
        payload = {
        "textQuery": query,   # Use "textQuery" instead of "query"
        "maxResultCount": 10
        }
        response = requests.post(url, headers=headers, json=payload)  # Use POST, not GET
        # if response.status_code != 200:
        #     return JsonResponse({"error": "Failed to fetch data from Google API"}, status=500)
        if response.status_code != 200:
            print("Google API Error:")
            print("Status Code:", response.status_code)
            print("Response Text:", response.text)
            return JsonResponse({"error": "Failed to fetch data from Google API"}, status=500)
        results = response.json()
        
        # Save the results to a JSON file
        os.makedirs(DATA_DIR, exist_ok=True)
        filename = clean_filename(query)
        filepath = os.path.join(DATA_DIR, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(
                {"query":query,
                 "results":results
                 }, f, ensure_ascii=False, indent=2)
        
        update_history(query,filename)
        return rag_answer(query)
        # # Filter results based on the query
        # filtered_results = [result for result in results if query.lower() in result['name'].lower()]
        # return JsonResponse(filtered_results, safe=False)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)