import json
import logging
import random
from typing import List, Dict, Any, Optional

import numpy as np
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

# Configure logging for better visibility
logger = logging.getLogger("milvus_client")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Global Configuration Constants ---
MILVUS_DB_FILE = "./milvus_data.db"  # Path to the local Milvus Lite database file
COLLECTION_NAME = "knowledge_base_milvus_lite" # Renamed for clarity to distinguish from server connections
EMBEDDING_DIM = 1536 # Dimension of your vector embeddings

# Field names for collection schema
ID_FIELD = "id"
SOURCE_DOC_FIELD = "source_document"
CHUNK_TEXT_FIELD = "chunk_text"
METADATA_FIELD = "metadata"
EMBEDDING_FIELD = "embedding"


def initialize_milvus_client(db_file: str = MILVUS_DB_FILE) -> MilvusClient:
    """
    Initializes and returns a Milvus Lite client.
    This client manages the local database file directly,
    eliminating the need for a separate Milvus server.
    """
    try:
        # MilvusClient with a file path automatically uses Milvus Lite
        client = MilvusClient(db_file)
        logger.info(f"Initialized Milvus Lite client using database file: {db_file}")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Milvus Lite client: {e}")
        raise # Re-raise the exception after logging


def define_collection_schema() -> CollectionSchema:
    """
    Defines the schema for the Milvus collection.
    This schema dictates the fields and their data types within the collection.
    """
    fields = [
        # Primary key field: automatically generated integer IDs
        FieldSchema(name=ID_FIELD, dtype=DataType.INT64, is_primary=True, auto_id=True),
        # Source document name, max length 1024 characters
        FieldSchema(name=SOURCE_DOC_FIELD, dtype=DataType.VARCHAR, max_length=1024),
        # Text content of the chunk, max length 8192 characters
        FieldSchema(name=CHUNK_TEXT_FIELD, dtype=DataType.VARCHAR, max_length=8192),
        # Metadata associated with the chunk, stored as a JSON string for flexibility
        FieldSchema(name=METADATA_FIELD, dtype=DataType.VARCHAR, max_length=4096),
        # Vector embedding field, essential for similarity search
        FieldSchema(name=EMBEDDING_FIELD, dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    # Create the CollectionSchema object
    return CollectionSchema(fields, description="Knowledge base chunks and embeddings for Milvus Lite")


def create_and_configure_collection(
    client: MilvusClient,
    collection_name: str,
    schema: CollectionSchema,
    embedding_field_name: str,
    metric_type: str = "IP" # Inner Product (IP) is a common metric for normalized embeddings
) -> None:
    """
    Creates the Milvus collection if it does not exist.
    If it exists, it will be dropped and recreated to ensure a clean state.
    An index is also created on the embedding field for efficient similarity search.

    Args:
        client: The initialized MilvusClient instance.
        collection_name: The name of the collection to create or reset.
        schema: The CollectionSchema object defining the collection structure.
        embedding_field_name: The name of the vector field in the schema to index.
        metric_type: The metric type for the index (e.g., "IP" for Inner Product, "L2" for Euclidean).
    """
    try:
        # Drop the collection if it already exists for a clean slate
        if client.has_collection(collection_name=collection_name):
            logger.info(f"Dropping existing collection '{collection_name}' for a clean start.")
            client.drop_collection(collection_name=collection_name)

        # Define index parameters. HNSW is a good general-purpose index.
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=embedding_field_name,
            index_type="HNSW",  # Hierarchical Navigable Small World index
            metric_type=metric_type,
            params={"M": 8, "efConstruction": 64} # Recommended parameters for HNSW
        )

        # Create the collection with the defined schema and index
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        logger.info(f"Created collection '{collection_name}' with schema and HNSW index.")
    except Exception as e:
        logger.error(f"Failed to create or configure collection '{collection_name}': {e}")
        raise # Re-raise the exception after logging


def insert_data(
    client: MilvusClient,
    collection_name: str,
    data_records: List[Dict[str, Any]]
) -> None:
    """
    Inserts data records (list of dictionaries) into the specified Milvus Lite collection.
    Each dictionary in 'data_records' represents an entity (row) in the collection.
    Flushes the collection after insertion to ensure data persistence.

    Args:
        client: The initialized MilvusClient instance.
        collection_name: The name of the collection to insert data into.
        data_records: A list of dictionaries, where each dict corresponds to an entity
                      and its keys match the collection's field names.
    """
    if not data_records:
        logger.warning("No data records provided for insertion.")
        return

    try:
        # MilvusClient.insert expects data as a list of dictionaries (row-major format)
        res = client.insert(collection_name=collection_name, data=data_records)
        # Flush the collection to ensure data is written to disk and available for search
        client.flush(collection_name=collection_name)
        logger.info(f"Inserted and flushed {len(data_records)} records into '{collection_name}'. "
                    f"Insert IDs: {res.get('ids', 'N/A')}")
    except Exception as e:
        logger.error(f"Failed to insert data into Milvus Lite collection '{collection_name}': {e}")
        raise # Re-raise the exception after logging


def search_collection(
    client: MilvusClient,
    collection_name: str,
    query_vectors: List[List[float]],
    top_k: int = 5,
    output_fields: Optional[List[str]] = None,
    # Additional search parameters can be passed as per Milvus documentation
    search_params: Optional[Dict[str, Any]] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Searches for the most similar vectors in the specified Milvus Lite collection.
    The collection is loaded into memory before performing the search.

    Args:
        client: The initialized MilvusClient instance.
        collection_name: The name of the collection to search.
        query_vectors: A list of embedding vectors to search with.
        top_k: The number of top-most similar results to return.
        output_fields: A list of field names to retrieve along with search results.
        search_params: Optional dictionary of search parameters (e.g., {"nprobe": 10} for HNSW).

    Returns:
        A list of dictionaries, where each dict contains information about a hit
        (e.g., 'id', 'distance', and requested 'output_fields'). Returns None on failure.
    """
    try:
        # Load the collection into memory before searching.
        # This is a mandatory step for search operations.
        client.load_collection(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' loaded for search.")

        # Default search parameters if not provided
        if search_params is None:
            search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

        # Perform the search
        # Data should be a list of query vectors
        results = client.search(
            collection_name=collection_name,
            data=query_vectors, # MilvusClient.search expects List[List[float]] directly
            limit=top_k,
            output_fields=output_fields or [], # Ensure it's a list even if None is passed
            search_params=search_params, # Pass the search parameters
        )

        # The results object is a list of lists of hits. For a single query, we take results[0].
        hits = []
        if results and results[0]:
            for hit in results[0]:
                hit_data = {
                    "id": hit.id,
                    "distance": hit.distance
                }
                # Add output_fields to the hit data
                if hit.entity:
                    for field in output_fields or []:
                        hit_data[field] = hit.entity.get(field)
                hits.append(hit_data)

        logger.info(f"Search returned {len(hits)} results.")
        return hits

    except Exception as e:
        logger.error(f"Milvus Lite search failed for collection '{collection_name}': {e}")
        return None
    finally:
        # It's good practice to release the collection from memory if not needed immediately
        # client.release_collection(collection_name=collection_name)
        # logger.info(f"Collection '{collection_name}' released from memory.")
        pass # Keeping it loaded for potential subsequent operations in a real app


def main():
    """
    Demonstrates the end-to-end usage of Milvus Lite with the refactored functions.
    This function simulates document chunking, embedding, insertion, and searching.
    """
    logger.info("--- Starting Milvus Lite Demonstration ---")

    # 1. Initialize Milvus Client (connects to local file)
    milvus_client = initialize_milvus_client()

    # 2. Define Collection Schema
    collection_schema = define_collection_schema()

    # 3. Create and Configure Collection (including index creation)
    # This will drop and recreate the collection for a clean start if it exists
    create_and_configure_collection(milvus_client, COLLECTION_NAME, collection_schema, EMBEDDING_FIELD)

    # --- Prepare Example Data ---
    # In a real application, you would generate these from your documents.
    # We use a dummy Document-like structure for demonstration.
    class DummyDocument:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata if metadata is not None else {}

    source_doc_name = "example_document.pdf"
    chunks = [
        DummyDocument(
            page_content="Artificial intelligence (AI) is intelligence demonstrated by machines.",
            metadata={"page": 1, "section": "intro"}
        ),
        DummyDocument(
            page_content="Machine learning (ML) is a subset of AI that enables systems to learn from data.",
            metadata={"page": 2, "section": "concepts"}
        ),
        DummyDocument(
            page_content="Deep learning is a subfield of machine learning inspired by the human brain.",
            metadata={"page": 3, "section": "concepts"}
        ),
        DummyDocument(
            page_content="Natural Language Processing (NLP) deals with the interaction between computers and human language.",
            metadata={"page": 4, "section": "applications"}
        ),
    ]

    # Generate dummy embeddings for demonstration.
    # In a real scenario, use an embedding model (e.g., from transformers library).
    embeddings = [np.random.rand(EMBEDDING_DIM).astype(np.float32).tolist() for _ in chunks]

    # Prepare data in the row-major format expected by MilvusClient.insert
    # Each dictionary represents one entity (row)
    data_records_to_insert = []
    for i, chunk in enumerate(chunks):
        data_records_to_insert.append({
            ID_FIELD: i, # Optional if auto_id=True, Milvus will generate
            SOURCE_DOC_FIELD: source_doc_name,
            CHUNK_TEXT_FIELD: chunk.page_content,
            METADATA_FIELD: json.dumps(chunk.metadata), # Convert metadata dict to JSON string
            EMBEDDING_FIELD: embeddings[i]
        })
    logger.info(f"Prepared {len(data_records_to_insert)} data records for insertion.")

    # 4. Insert Data
    insert_data(milvus_client, COLLECTION_NAME, data_records_to_insert)

    # 5. Perform a Search
    query_text = "What is deep learning?"
    # Generate a dummy query embedding for the search
    query_embedding = [np.random.rand(EMBEDDING_DIM).astype(np.float32).tolist()]

    search_results = search_collection(
        milvus_client,
        COLLECTION_NAME,
        query_vectors=query_embedding,
        top_k=2,
        output_fields=[CHUNK_TEXT_FIELD, SOURCE_DOC_FIELD, METADATA_FIELD]
    )

    print("\n--- Search Results ---")
    if search_results:
        for i, res in enumerate(search_results):
            print(f"Result {i+1}:")
            print(f"  ID: {res.get(ID_FIELD)}")
            print(f"  Distance: {res.get('distance'):.4f}")
            print(f"  Chunk Text: {res.get(CHUNK_TEXT_FIELD)}")
            print(f"  Source Document: {res.get(SOURCE_DOC_FIELD)}")
            print(f"  Metadata: {res.get(METADATA_FIELD)}")
            print("-" * 20)
    else:
        print("No search results found or an error occurred.")

    logger.info("--- Milvus Lite Demonstration Completed ---")

if __name__ == "__main__":
    main()
