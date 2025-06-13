"""Main script for document processing pipeline."""
import os
import argparse
from config.settings import (
    MARKDOWN_PATH, OUTPUT_DIR, DEFAULT_OUTPUT_FILENAME,
    CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS, KEEP_SEPARATOR,
    TOKENIZER_TYPE, TOKENIZER_MODEL, MAX_TOKENS
)
from processors.document_processor import load_document, create_text_splitter
from utils.url_extractor import extract_urls
from utils.logger import setup_logger, log_processing_steps
from utils.tokenizer import Tokenizer
from utils.embedder import Embedder
from db import OracleLoader

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process markdown document and extract URLs.')
    parser.add_argument('--output', '-o', 
                       default=DEFAULT_OUTPUT_FILENAME,
                       help='Output filename (default: main_out.txt)')
    parser.add_argument('--load-to-oracle', action='store_true',
                       help='Load processed chunks and embeddings to Oracle database')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for Oracle inserts (default: 100)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up output path
    output_filename = args.output
    if not output_filename.endswith('.txt'):
        output_filename += '.txt'
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Set up logging
    logger = setup_logger()
    
    # Initialize tokenizer and embedder
    tokenizer = Tokenizer(TOKENIZER_TYPE, TOKENIZER_MODEL)
    embedder = Embedder(TOKENIZER_TYPE, TOKENIZER_MODEL, tokenizer, MAX_TOKENS)
    
    # Load and process document
    docs = load_document(MARKDOWN_PATH)
    
    # Create and use text splitter
    text_splitter = create_text_splitter(
        CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS, KEEP_SEPARATOR
    )
    chunks = text_splitter.split_documents(docs)
    
    # Log processing steps
    log_processing_steps(logger, MARKDOWN_PATH, chunks)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Extract text content from chunks
    chunk_texts = [chunk.page_content for chunk in chunks]
    
    # Generate embeddings
    embeddings = embedder.embed_chunks(chunk_texts)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Document split into {len(chunks)} chunks\n\n")
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            text = chunk.page_content
            token_count = tokenizer.count_tokens(text)
            
            f.write(f"=== Chunk {i+1} ===\n")
            f.write(f"Character Length: {len(text)}\n")
            f.write(f"Token Count: {token_count}\n")
            
            urls = extract_urls(text)
            if urls:
                f.write("URLs found in chunk:\n")
                for url in urls:
                    f.write(f"- {url}\n")
            
            f.write("\nContent:\n")
            f.write(text)
            f.write("\n\n")
            
            # Write embedding information
            f.write(f"=== Embedding {i+1} ===\n")
            f.write(f"Vector Length: {len(embedding)}\n")
            f.write("Values:\n")
            f.write(",".join(map(str, embedding)))
            f.write("\n\n")
    
    logger.info(f"Results written to {output_path}")
    
    # Load to Oracle if requested
    if args.load_to_oracle:
        try:
            loader = OracleLoader()
            rows_inserted = loader.insert_data(
                source_doc_name=os.path.basename(MARKDOWN_PATH),
                chunks=chunks,
                embeddings=embeddings,
                batch_size=args.batch_size
            )
            
            if rows_inserted is not None:
                logger.info(f"Successfully loaded {rows_inserted} chunks to Oracle database")
            else:
                logger.error("Failed to load data to Oracle database")
        except Exception as e:
            logger.error(f"Error during Oracle loading: {e}")

if __name__ == "__main__":
    main()
