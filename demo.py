# This script adds Azure OpenAI to generate
# a query vector and then performs a search.

# Ensure you have the necessary libraries installed:
# pip install pymongo openai python-dotenv

import pymongo
import os
import time
import openai
from dotenv import load_dotenv
from pymongo.errors import OperationFailure, ConnectionFailure, ConfigurationError
from pymongo.operations import SearchIndexModel

# --- Configuration ---
# Load environment variables from a .env file for security
load_dotenv()

ATLAS_CONNECTION_STRING = "mongodb://localhost:27017/?retryWrites=true&w=majority&directConnection=true"
DB_NAME = "sample_mflix"
COLLECTION_NAME = "embedded_movies"
INDEX_NAME = "vector_index_plot"
VECTOR_FIELD_PATH = "plot_embedding"
VECTOR_DIMENSIONS = 1536  # Dimensions for OpenAI's text-embedding-ada-002

def generate_embedding(text: str, client: openai.AzureOpenAI) -> list[float]:
    """
    Generates a vector embedding for a given text using Azure OpenAI.
    """
    print(f"[INFO] Generating embedding for query: '{text}'")
    try:
        # Note: The 'model' parameter for Azure OpenAI refers to your deployment name.
        deployment_name = "text-embedding-ada-002"  # Change this to your deployment name
        if not deployment_name:
            raise ValueError("[FATAL] AZURE_OAI_DEPLOYMENT environment variable not set.")
            
        response = client.embeddings.create(input=[text], model=deployment_name)
        return response.data[0].embedding
    except Exception as e:
        print(f"[ERROR] Failed to generate embedding: {e}")
        return None

def main():
    """
    Main function to connect to MongoDB, manage vector indexes, and run a vector search.
    """
    print("--- MongoDB & Azure OpenAI Vector Search Example (2025) ---")

    mongo_client = None
    try:
        # 1. Connect to MongoDB Atlas
        print("\n[INFO] Connecting to MongoDB Atlas cluster...")
        if not ATLAS_CONNECTION_STRING or "<username>" in ATLAS_CONNECTION_STRING:
            print("[FATAL] Please set your ATLAS_CONNECTION_STRING in the .env file.")
            return

        mongo_client = pymongo.MongoClient(ATLAS_CONNECTION_STRING)
        mongo_client.admin.command('ping')
        print("[SUCCESS] MongoDB connection successful.")

        db = mongo_client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # 2. Define the Vector Search Index
        print("[INFO] Defining Vector Search index model...")
        search_index_model = SearchIndexModel(
            name=INDEX_NAME,
            type="vectorSearch",
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": VECTOR_FIELD_PATH,
                        "numDimensions": VECTOR_DIMENSIONS,
                        "similarity": "cosine",
                    }
                ]
            }
        )

        # 3. Create the Vector Search Index (if it doesn't exist)
        print(f"\n[ACTION] Checking/Creating index '{INDEX_NAME}'...")
        try:
            # Check for existing index first to avoid unnecessary waits
            existing_indexes = [idx['name'] for idx in collection.list_search_indexes()]
            if INDEX_NAME in existing_indexes:
                 print(f"[WARN] Index '{INDEX_NAME}' already exists. Skipping creation.")
            else:
                collection.create_search_index(model=search_index_model)
                print(f"[SUCCESS] Index creation command sent for '{INDEX_NAME}'. Monitoring status...")
                # Monitor index build status
                while True:
                    index_status = list(collection.list_search_indexes(name=INDEX_NAME))
                    if not index_status:
                        print("[INFO] Waiting for index creation to initialize...")
                        time.sleep(10)
                        continue
                    status = index_status[0].get('status')
                    print(f"[INFO] Current index status: {status}")
                    if status == 'READY':
                        print("[SUCCESS] Index is built and ready for use. ✅")
                        break
                    elif status == 'FAILED':
                        print("[ERROR] Index creation failed. Check the Atlas UI for details.")
                        return
                    time.sleep(10)
        except OperationFailure as e:
            # Handle cases where the index was created between the check and the create call
            if "already exists" in str(e):
                print(f"[WARN] Index '{INDEX_NAME}' already exists.")
            else:
                print(f"[ERROR] An error occurred: {e}")
                return

        # 4. Perform a Vector Search using Azure OpenAI
        user_input = input(f"\n[PROMPT] Do you want to perform a vector search? (yes/no): ").lower()
        if user_input == 'yes':
            print("\n[ACTION] Initializing Azure OpenAI client...")
            try:
                # Setup Azure OpenAI client
                azure_oai_client = openai.AzureOpenAI(
                    api_key="",
                    api_version="2023-05-15", # A common, stable version
                    azure_endpoint="https://.openai.azure.com"
                )
                
                query_text = "A tale of redemption and friendship in a prison."
                query_vector = generate_embedding(query_text, azure_oai_client)

                if query_vector:
                    print("[INFO] Executing $vectorSearch aggregation...")
                    pipeline = [
                        {
                            "$vectorSearch": {
                                "index": INDEX_NAME,
                                "path": VECTOR_FIELD_PATH,
                                "queryVector": query_vector,
                                "numCandidates": 150, # Number of candidates to consider
                                "limit": 5, # Number of results to return
                            }
                        },
                        {
                            "$project": {
                                "title": 1,
                                "plot": 1,
                                "score": { "$meta": "vectorSearchScore" } # Include the search score
                            }
                        }
                    ]
                    results = collection.aggregate(pipeline)
                    print("\n--- Search Results ---")
                    for doc in results:
                        print(f"  🎬 Title: {doc['title']}\n"
                              f"     Score: {doc['score']:.4f}\n"
                              f"     Plot: {doc['plot'][:150]}...\n")

            except Exception as e:
                print(f"[FATAL] An error occurred during the search operation: {e}")

        # 5. List and Drop Index
        print(f"\n[INFO] Current search indexes for '{COLLECTION_NAME}':")
        for idx in collection.list_search_indexes():
            print(f"  - Name: {idx['name']}, Status: {idx.get('status', 'N/A')}")
        
        user_input = input(f"\n[PROMPT] Do you want to drop the index '{INDEX_NAME}'? (yes/no): ").lower()
        if user_input == 'yes':
            print(f"\n[ACTION] Dropping index '{INDEX_NAME}'...")
            collection.drop_search_index(INDEX_NAME)
            print(f"[SUCCESS] Index '{INDEX_NAME}' dropped.")
        else:
            print("\n[INFO] Skipping index drop.")

    except ConfigurationError as e:
        print(f"[FATAL] Configuration error: {e}")
    except ConnectionFailure as e:
        print(f"[FATAL] Connection failed: {e}")
    except Exception as e:
        print(f"[FATAL] An unexpected error occurred: {e}")
    finally:
        if mongo_client:
            mongo_client.close()
            print("\n[INFO] MongoDB connection closed.")

if __name__ == "__main__":
    main()
