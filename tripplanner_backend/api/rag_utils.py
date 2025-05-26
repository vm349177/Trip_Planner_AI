import os
import json
import requests
import torch
# from langchain_community.llms import Ollama
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

max_seq_length = 2048  # or 4096
dtype = torch.float16  # or bfloat16 if you're on CPU or low-memory GPU
api_key = os.getenv("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/chat/completions"
DATA_DIR = 'tripplanner_backend/data'
HISTORY_FILE = os.path.join(DATA_DIR, 'history.json')
VECTOR_DIR = 'tripplanner_backend/vectorstore'
# print("GOOGLE_API_KEY =", os.getenv("GOOGLE_API_KEY"))
# embedding_fn=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_fn=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=VECTOR_DIR,
)
vectorstore=Chroma(
    embedding_function=embedding_fn,
    persist_directory=VECTOR_DIR,
    collection_name="trip_planner",
    # embedding_dimension=embedding_fn.get_sentence_embedding_dimension(),
)
# llm=Ollama(model="llama3:8b-instruct-q4_0", temperature=0.7)

llm = ChatGroq(
            model="llama3-8b-8192",  # or "mixtral-8x7b-32768" etc.
            temperature=0.7,
            groq_api_key=api_key,
        )
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def get_recent_context(limit=5):
    """Reads the last 5 queries and summarizes the data for RAG context."""
    if not os.path.exists(HISTORY_FILE):
        return ""

    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)[-limit:]

    context = ""
    for entry in history:
        file_path = os.path.join(DATA_DIR, entry['file'])
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                places = data.get("results", [])
                summary = "\n".join([
                    f"- {p['displayName']['text']} ({p.get('formattedAddress', 'No address')})"
                    for p in places[:3]  # limit to 3 places for brevity
                ])
                context += f"\nQuery: {entry['query']}\n{summary}\n"
    return context.strip()

def run_rag(query):
    try:
        filename = f"{query.replace(' ', '_')}.json"
        file_path = os.path.join(DATA_DIR, filename)

        with open(file_path, 'r',encoding='utf-8') as f:
            data =json.load(f)

        results = data.get("results", {}).get("places", [])
        if not results:
            return {"answer": "No results found for the query."}
        # Create documents for each place
        documents = []
        for place in results:
            name = place.get("displayName", {}).get("text", "")
            address = place.get("formattedAddress",{})
            types = ", ".join(place.get("types", []))
            location = place.get("location", {})
            description = place.get("editorialSummary",{}).get("text", "")
            
            content = f"Name: {name}\nAddress: {address}\nTypes: {types}\nLocation: {location}\nDescription: {description}"
            documents.append(Document(
                page_content=content,
                metadata={"name": name}))
        
        docs = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_fn,
            persist_directory=VECTOR_DIR,
            collection_name="trip_planner",
            # embedding_dimension=embedding_fn.get_dimension(),
        )

        # vectorstore.persist()
        retriver = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa=RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriver,
            return_source_documents=True,
        )

        results = qa.invoke({"query": query})
        return {
            "answer": results['result'],
            "source_documents": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results['source_documents']
            ]
        }
    except Exception as e:
        print(f"Error in run_rag: {e}")
        return {"error": "Failed to get answer from RAG"}
    
# rag_utils.py (continued)

def run_rag_history(query, similar_query_obj):
    try:
        # If similar_query_obj is a dict (from history), get the original query
        similar_query = similar_query_obj.get('query') if isinstance(similar_query_obj, dict) else similar_query_obj

        # Clean filename
        filename = similar_query.replace(" ", "_")
        filename = ''.join(e for e in filename if e.isalnum() or e == '_') + ".json"
        filepath = os.path.join(DATA_DIR, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = data.get("results", {}).get("places", [])
        if not results:
            return {"answer": "No places found in the historical query data."}

        # Prepare documents
        documents = []
        for place in results:
            name = place.get("displayName", {}).get("text", "")
            address = place.get("formattedAddress", {})
            types = ", ".join(place.get("types", []))
            location = place.get("location", {})
            description = place.get("editorialSummary", {}).get("text", "")
            content = f"Name: {name}\nAddress: {address}\nTypes: {types}\nLocation: {location}\nDescription: {description}"
            documents.append(Document(page_content=content, metadata={"name": name}))

        # Split if necessary
        docs = text_splitter.split_documents(documents)

        # Create a fresh ChromaDB in memory (since it's past data)
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding_fn,
            persist_directory=None  # No need to persist history-based temporary retrieval
        )

        retriever = vectordb.as_retriever()
        prompt_template = PromptTemplate(
            input_variables=["context", "query", "similar"],
            template="""
        You are a travel assistant AI.

        You have already handled a similar user query:
        "{similar}"

        Now the user has a new query:
        "{query}"

        Use the following context, which contains travel-related place data in JSON format, to answer the new query based on your prior understanding of the similar query.

        Context:
        {context}

        Give a helpful, focused, and accurate answer related to the query.
        """
        )

        # Create the QA chain
        # llm = Ollama(model="llama3")

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )

        result = qa.invoke({"query": query,"similar": similar_query})
        return {"answer": result["result"],
                "source_documents": [
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result['source_documents']
                ]
            }

    except Exception as e:
        print(f"[RAG HISTORY ERROR] {e}")
        return {"error": str(e)}
