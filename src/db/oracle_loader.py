"""Oracle database operations for document processing pipeline."""
import oracledb
import os
import json
from typing import List, Optional
from langchain.schema import Document
from dotenv import load_dotenv
from utils.logger import setup_logger

# Load environment variables
load_dotenv()

# Database configuration
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_DSN = os.environ.get("DB_DSN")

# SQL statements
INSERT_CHUNK_SQL = """
    INSERT INTO knowledge_base (source_document, chunk_text, metadata, embedding)
    VALUES (:1, :2, :3, :4)
"""

class OracleLoader:
    """Handles loading document chunks and embeddings into Oracle database."""
    
    def __init__(self):
        """Initialize the Oracle loader with database configuration."""
        self.logger = setup_logger()
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate that all required database configuration is present."""
        if not all([DB_USER, DB_PASSWORD, DB_DSN]):
            raise ValueError(
                "Missing database configuration. Please set DB_USER, DB_PASSWORD, "
                "and DB_DSN environment variables."
            )
    
    def _prepare_batch_data(
        self, 
        source_doc_name: str, 
        chunks: List[Document], 
        embeddings: List[List[float]]
    ) -> List[tuple]:
        """
        Prepare data for batch insertion.
        
        Args:
            source_doc_name: Name of the source document
            chunks: List of Document objects
            embeddings: List of embedding vectors
            
        Returns:
            List of tuples ready for batch insertion
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
            
        return [
            (
                source_doc_name,
                chunk.page_content,
                json.dumps(chunk.metadata),
                embedding
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
    
    def insert_data(
        self,
        source_doc_name: str,
        chunks: List[Document],
        embeddings: List[List[float]],
        batch_size: int = 100
    ) -> Optional[int]:
        """
        Insert document chunks and embeddings into Oracle database.
        
        Args:
            source_doc_name: Name of the source document
            chunks: List of Document objects
            embeddings: List of embedding vectors
            batch_size: Number of records to insert in each batch
            
        Returns:
            Number of rows inserted, or None if insertion failed
        """
        try:
            data_to_insert = self._prepare_batch_data(source_doc_name, chunks, embeddings)
            total_rows = 0
            
            with oracledb.connect(
                user=DB_USER,
                password=DB_PASSWORD,
                dsn=DB_DSN
            ) as connection:
                with connection.cursor() as cursor:
                    
                    # === THE FIX IS HERE ===
                    # Pre-define the data types for the columns, especially the VECTOR type.
                    # This must be done before cursor.executemany().
                    cursor.setinputsizes(
                        None,                   # :1 source_document (VARCHAR2, can be inferred)
                        oracledb.DB_TYPE_CLOB,  # :2 chunk_text
                        oracledb.DB_TYPE_JSON,  # :3 metadata
                        oracledb.DB_TYPE_VECTOR # :4 embedding
                    )
                    # =======================

                    # Process in batches
                    for i in range(0, len(data_to_insert), batch_size):
                        batch = data_to_insert[i:i + batch_size]
                        cursor.executemany(INSERT_CHUNK_SQL, batch)
                        total_rows += cursor.rowcount
                        
                        # Log progress
                        self.logger.info(
                            f"Inserted batch of {len(batch)} rows "
                            f"({i + len(batch)}/{len(data_to_insert)} total)"
                        )
                    
                    connection.commit()
                    
            self.logger.info(f"Successfully inserted {total_rows} rows for {source_doc_name}")
            return total_rows
            
        except oracledb.DatabaseError as e:
            self.logger.error(f"Database error during insertion: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during insertion: {e}")
            return None