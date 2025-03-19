import os
from pdf_processing import extract_text_from_pdfs
from chunking import create_chunks
from vector_store import create_vector_store
from retriever import load_faiss_index, retrieve
from transformers import pipeline
import textwrap

def generate_text_with_hf(context, max_length=1024):
    """Generate text using a Hugging Face model based on the provided context."""
    generator = pipeline('text-generation', model='distilgpt2', tokenizer='distilgpt2')  # Use smaller model for efficiency
    
    # Convert context to string if it's not already
    if isinstance(context, list):
        context = " ".join(context)

    # Truncate context to the max_length allowed by the model
    truncated_context = context[:max_length]

    # Generate the text using the context
    response = generator(truncated_context, max_new_tokens=50, num_return_sequences=1)
    return response[0]['generated_text']

def format_output(text, width=60):
    """Split the text into multiple lines with a specific width."""
    return "\n".join(textwrap.wrap(text, width))

def main():
    # Paths to directories
    pdf_folder = '/content/drive/MyDrive/LawGPT/dataset/pdfs'
    chunk_folder = '/content/drive/MyDrive/LawGPT/dataset/chunks'
    index_path = '/content/drive/MyDrive/LawGPT/vector_store.index'
    chunk_paths_file = '/content/drive/MyDrive/LawGPT/chunk_paths.txt'

    # Step 4: Retrieve relevant chunks for a given query
    print("Retrieving relevant chunks...")
    query = input("Enter your law query:")  # Replace with your query
    top_k = 5  # Number of top results to retrieve

    index, chunk_paths = load_faiss_index(index_path, chunk_paths_file)
    retrieved_chunks = retrieve(query, index, chunk_paths, top_k=top_k)

    print("Retrieved Chunks:")
    combined_context = ""
    for i, chunk in enumerate(retrieved_chunks):
        combined_context += chunk + " "  # Combine chunks for context

    # Step 5: Generate a response based on the retrieved chunks
    print("\nGenerating response based on retrieved context...")
    response = generate_text_with_hf(combined_context)

    # Step 6: Print the answer in 3 or 4 lines
    formatted_response = format_output(response)
    print("\nAnswer: ")
    print(formatted_response)

if __name__ == "__main__":
    main()
