import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_faiss_index(index_path, chunk_paths_file):
    """
    Loads the FAISS index and corresponding chunk paths from disk.
    
    Args:
        index_path (str): Path to the FAISS index file.
        chunk_paths_file (str): Path to the text file containing chunk paths.
    
    Returns:
        index (faiss.Index): The loaded FAISS index.
        chunk_paths (list): List of paths to the chunks.
    """
    # Load the FAISS index
    index = faiss.read_index(index_path)
    
    # Load the chunk paths
    with open(chunk_paths_file, 'r') as f:
        chunk_paths = f.read().splitlines()
    
    return index, chunk_paths

def retrieve(query, index, chunk_paths, model_name='sentence-transformers/all-MiniLM-L6-v2', top_k=3):
    """
    Retrieves the most relevant chunks from the FAISS index for a given query.
    
    Args:
        query (str): The query string.
        index (faiss.Index): The FAISS index.
        chunk_paths (list): List of paths to the chunks.
        model_name (str): Name of the sentence transformer model.
        top_k (int): Number of top results to return.
    
    Returns:
        retrieved_chunks (list): List of retrieved text chunks.
    """
    model = SentenceTransformer(model_name)
    query_vector = model.encode([query])[0].astype('float32')
    
    # Search the index
    distances, indices = index.search(np.array([query_vector]), top_k)
    
    retrieved_chunks = []
    for idx in indices[0]:
        if idx != -1:  # Valid index
            with open(chunk_paths[idx], 'r') as f:
                retrieved_chunks.append(f.read())
    
    return retrieved_chunks

if __name__ == "__main__":
    # Example usage for testing
    index, chunk_paths = load_faiss_index('/content/drive/MyDrive/LawGPT/vector_store.index', '/content/drive/MyDrive/LawGPT/chunk_paths.txt')
    query = "what if person a is blackmailed by person b"
    top_k = 3
    results = retrieve(query, index, chunk_paths, top_k=top_k)
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---\n")
        print(result)
