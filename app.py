import streamlit as st
import os
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from dotenv import load_dotenv
import fitz  # PyMuPDF
from openai import OpenAI
import re

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Elasticsearch client
es = Elasticsearch(["http://localhost:9200"])


# Initialize Sentence Transformer model
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()


def create_index_if_not_exists(index_name: str):
    """
    Create an Elasticsearch index if it doesn't already exist.

    This function checks if the specified index exists in Elasticsearch. If it doesn't,
    it creates a new index with a predefined mapping structure optimized for storing
    and searching PDF chunks.

    Args:
        index_name (str): The name of the index to create or check.

    The created index includes fields for:
    - paper_id: Unique identifier for each paper (keyword for exact matching)
    - chunk_id: Numerical ID for each chunk within a paper
    - title and text: Full-text searchable fields
    - embedding: Dense vector for similarity search (384-dimensional, cosine similarity)
    - metadata: Nested object with searchable paper metadata

    This structure supports full-text search, vector similarity, and structured querying.
    """
    if not es.indices.exists(index=index_name):
        print(f"Creating new index '{index_name}'...")
        mapping = {
            'mappings': {
                'properties': {
                    'paper_id': {'type': 'keyword'},
                    'chunk_id': {'type': 'integer'},
                    'title': {'type': 'text'},
                    'text': {'type': 'text'},
                    'embedding': {
                        'type': 'dense_vector',
                        'dims': 384,
                        'index': True,
                        'similarity': 'cosine'
                    },
                    'metadata': {
                        'properties': {
                            'title': {'type': 'text'},
                            'author': {'type': 'text'},
                            'subject': {'type': 'text'},
                            'keywords': {'type': 'text'},
                            'creation_date': {'type': 'date'}
                        }
                    }
                }
            }
        }
        es.indices.create(index=index_name, body=mapping)
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text content from a PDF file.

    This function opens a PDF file specified by the file path, iterates through
    all pages, and extracts the text content from each page. The extracted text
    from all pages is concatenated into a single string.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: A string containing all the extracted text from the PDF.

    Note:
        This function uses the fitz library (PyMuPDF) to handle PDF operations.
    """
    with fitz.open(file_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def clean_text(text: str) -> str:
    """
    Clean and normalize the input text.

    This function performs two main operations:
    1. Removes extra whitespace, including leading and trailing spaces.
    2. Removes special characters, keeping only alphanumeric characters, spaces, and basic punctuation.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned and normalized text.

    Example:
        >>> clean_text("  Hello,   World!  @#$%  ")
        "Hello, World!"
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s.,;!?]', '', text)
    return text

def extract_metadata(file_path: str) -> dict:
    """
    Extracts metadata from a PDF file.

    This function opens a PDF file specified by the file path and extracts
    its metadata, including title, author, subject, keywords, and creation date.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        dict: A dictionary containing the extracted metadata with the following keys:
            - 'title': The title of the PDF (empty string if not available)
            - 'author': The author of the PDF (empty string if not available)
            - 'subject': The subject of the PDF (empty string if not available)
            - 'keywords': The keywords associated with the PDF (empty string if not available)
            - 'creation_date': The creation date of the PDF (empty string if not available)

    Note:
        This function uses the fitz library (PyMuPDF) to handle PDF operations.
    """
    with fitz.open(file_path) as doc:
        metadata = doc.metadata
    return {
        'title': metadata.get('title', ''),
        'author': metadata.get('author', ''),
        'subject': metadata.get('subject', ''),
        'keywords': metadata.get('keywords', ''),
        'creation_date': metadata.get('creationDate', ''),
    }


def chunk_text(text: str, max_length: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits a given text into chunks of specified maximum length with overlap.

    This function takes a long text and divides it into smaller chunks, ensuring that
    each chunk does not exceed the specified maximum length. It also allows for an
    overlap between chunks to maintain context continuity.

    Args:
        text (str): The input text to be chunked.
        max_length (int, optional): The maximum length of each chunk. Defaults to 500.
        overlap (int, optional): The number of words to overlap between chunks. Defaults to 50.

    Returns:
        List[str]: A list of text chunks, where each chunk is a string of words.

    Note:
        The function splits the text into words and then reconstructs the chunks,
        so the actual length of each chunk may be slightly less than max_length
        to avoid splitting in the middle of a word.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(' '.join(current_chunk))
            overlap_words = current_chunk[-overlap:]
            current_chunk = overlap_words + [word]
            current_length = sum(len(w) + 1 for w in current_chunk)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def index_chunks(paper_id: str, title: str, chunks: List[str]):
    """
    Index chunks of text into Elasticsearch.

    This function takes chunks of text from a paper and indexes them into Elasticsearch.
    Each chunk is embedded, normalized, and added to a list of actions for bulk indexing.

    Args:
        paper_id (str): Unique identifier for the paper.
        title (str): Title of the paper.
        chunks (List[str]): List of text chunks to be indexed.

    Returns:
        None

    Raises:
        Exception: If there's an error during bulk indexing.

    Note:
        This function assumes that global 'embedder' and 'es' objects are available,
        representing the embedding model and Elasticsearch client respectively.
    """
    actions = []
    for idx, chunk in enumerate(chunks):
        try:
            embedding = embedder.encode(chunk)
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            # Create a document for Elasticsearch
            doc = {
                'paper_id': paper_id,
                'chunk_id': idx,
                'title': title,
                'text': chunk,
                'embedding': embedding.tolist()
            }
            # Create an action for bulk indexing
            action = {
                "_index": "paper_chunks",
                "_id": f"{paper_id}_{idx}",
                "_source": doc
            }
            actions.append(action)
        except Exception as e:
            logger.error(f"Error processing chunk {idx}: {e}")
    
    if not actions:
        logger.error("No valid actions to index. Check chunk processing.")
        return

    try:
        success, failed = bulk(es, actions)
        logger.info(f"Indexed {success} chunks, {failed} failed")
        if failed > 0:
            logger.warning("Some chunks failed to index. Check Elasticsearch logs for details.")
    except Exception as e:
        logger.error(f"Error during bulk indexing: {e}")

def check_index_status():
    """
    Check the status of the 'paper_chunks' index in Elasticsearch.

    This function retrieves statistics about the 'paper_chunks' index and prints
    the total number of documents in the index. If an error occurs during the
    process, it prints an error message.

    Returns:
        None

    Raises:
        Exception: If there's an error while retrieving index statistics.
            The error message is printed to the console.

    Note:
        This function assumes that a global 'es' object is available,
        representing the Elasticsearch client.
    """
    try:
        index_stats = es.indices.stats(index="paper_chunks")
        doc_count = index_stats['indices']['paper_chunks']['total']['docs']['count']
        print(f"Number of documents in index: {doc_count}")
    except Exception as e:
        print(f"Error checking index status: {e}")

def search_similar_chunks(query: str, top_k: int = 5):
    """
    Search for similar chunks in the Elasticsearch index based on a given query.

    This function encodes the input query, normalizes the embedding, and performs
    a similarity search using cosine similarity in the Elasticsearch index.

    Args:
        query (str): The input query to search for similar chunks.
        top_k (int, optional): The number of top similar chunks to return. Defaults to 5.

    Returns:
        list: A list of dictionaries containing the top_k most similar chunks,
              including their metadata and similarity scores.

    Prints:
        - The dimension of the query vector.
        - The first 5 elements of the query vector.
        - The number of hits (similar chunks) found.

    Note:
        This function assumes that global 'embedder' and 'es' objects are available,
        representing the sentence encoder and Elasticsearch client respectively.
    """

    query_embedding = embedder.encode(query)
    # Normalize the embedding
    query_vector = query_embedding / np.linalg.norm(query_embedding)
    
    print(f"Query vector dimension: {len(query_vector)}")
    print(f"Query vector: {query_vector[:5]}...")  # Print first 5 elements

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_vector.tolist()}
            }
        }
    }

    response = es.search(
        index="paper_chunks",
        body={
            "size": top_k,
            "query": script_query,
            "_source": ["paper_id", "chunk_id", "title", "text", "metadata"],
            "sort": [{"_score": "desc"}]
        }
    )
    
    print(f"Number of hits: {len(response['hits']['hits'])}")
    return response['hits']['hits']

def check_elasticsearch_connection():
    """
    Check the connection to Elasticsearch.

    This function attempts to ping the Elasticsearch server to verify the connection.
    It logs the result of the connection attempt and returns a boolean indicating
    whether the connection was successful.

    Returns:
        bool: True if the connection to Elasticsearch was successful, False otherwise.

    Logs:
        - Info message if the connection is successful.
        - Error message if the connection fails or an exception occurs.

    Note:
        This function assumes that a global 'es' object (Elasticsearch client) and
        'logger' object are available for making the connection and logging respectively.
    """
    try:
        if es.ping():
            logger.info("Successfully connected to Elasticsearch")
            return True
        else:
            logger.error("Could not connect to Elasticsearch")
            return False
    except Exception as e:
        logger.error(f"Error connecting to Elasticsearch: {e}")
        return False

def main():
    st.title("PDF Question Answering System")

    # Check Elasticsearch connection
    if not check_elasticsearch_connection():
        st.error("Failed to connect to Elasticsearch. Please check your connection and try again.")
        return


    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())

        # Extract text from PDF
        text = extract_text_from_pdf("temp.pdf")
        text = clean_text(text)
        st.success(f"Extracted {len(text)} characters from the PDF.")

        metadata = extract_metadata("temp.pdf")
        st.write(f"Metadata: {metadata}")

        # Chunk the text
        chunks = chunk_text(text, max_length=500)
        st.success(f"Created {len(chunks)} chunks from the text.")

        # Create index and index chunks
        create_index_if_not_exists('paper_chunks')
        paper_id = uploaded_file.name.split(".")[0]
        title = paper_id
        logger.info(f"Indexing chunks for paper: {paper_id}")
        index_chunks(paper_id, title, chunks)

        # Check index status after indexing
        check_index_status()
        
        # Verify that the chunks were indexed
        verify_query = {
            "query": {
                "term": {
                    "paper_id": paper_id
                }
            }
        }

        try:
            result = es.search(index="paper_chunks", body=verify_query)
            indexed_count = result['hits']['total']['value']
            logger.info(f"Number of chunks indexed for this paper: {indexed_count}")

            if indexed_count == 0:
                st.error("Failed to index chunks. Please check the application logs for more details.")
            else:
                st.success(f"Successfully indexed {indexed_count} chunks.")
        except Exception as e:
            logger.error(f"Error verifying indexed chunks: {e}")
            st.error("An error occurred while verifying indexed chunks. Please check the application logs.")

        # User query input
        user_query = st.text_input("Ask a question about the PDF:")

        # Add a slider for the number of results
        num_results = st.slider("Number of relevant excerpts to display", min_value=1, max_value=10, value=5)
        
        # show num_results
        st.write(f"Number of relevant excerpts to display: {num_results}")

        if user_query:
            # Search similar chunks
            similar_chunks = search_similar_chunks(user_query, top_k=num_results)

            st.write("number of similar chunks: ", len(similar_chunks))

            # Display similar chunks
            st.subheader("Relevant excerpts:")
            for i, hit in enumerate(similar_chunks):
                st.text_area(f"Excerpt {i+1}", hit['_source']['text'], height=100)

            # Generate answer using OpenAI
            try:
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

                relevant_texts = [hit['_source']['text'] for hit in similar_chunks]
                combined_text = "\n\n".join(relevant_texts)
                prompt = f"""You are an expert assistant. Based on the following excerpts from a research paper, answer the question concisely and accurately.

                Question: {user_query}

                Excerpts:
                {combined_text}

                Answer:"""

                response = client.chat.completions.create(
                    model="gpt-4-0125-preview",  # Make sure to use an available model
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.5,
                )
                answer = response.choices[0].message.content
                st.subheader("Generated Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error generating answer: {e}")

        # Clean up temporary file
        os.remove("temp.pdf")

if __name__ == "__main__":
    main()