import streamlit as st
from transformers import pipeline
from pdf_processing import extract_text_from_pdfs
from chunking import create_chunks
from vector_store import create_vector_store
from retriever import load_faiss_index, retrieve

# Function to load GPT-2 model
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

# Function to format text
def format_output(text, width=60):
    """Split the text into multiple lines with a specific width."""
    import textwrap
    return "\n".join(textwrap.wrap(text, width))

# Main Streamlit app function
def main():
    st.set_page_config(page_title="Legal Chatbot", page_icon="⚖️", layout="wide")

    # Sidebar for model settings
    st.sidebar.title("Chatbot Settings")
    max_length = st.sidebar.slider("Max response length", 50, 500, 150)

    # App title
    st.title("⚖️ Legal Chatbot")
    st.write("Ask any legal question related to Indian law.")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_input = st.text_input("Your question", key="input", placeholder="Ask a legal question...", help="Type your legal question and press Enter.")

    # Retrieve relevant chunks for a given query
    def retrieve_chunks(query):
        # Assuming these are defined elsewhere or replace with mock data
        index_path = '/content/drive/MyDrive/LawGPT/vector_store.index'
        chunk_paths_file = '/content/drive/MyDrive/LawGPT/chunk_paths.txt'
        index, chunk_paths = load_faiss_index(index_path, chunk_paths_file)
        top_k = 5  # Number of top results to retrieve
        retrieved_chunks = retrieve(query, index, chunk_paths, top_k=top_k)
        combined_context = " ".join(retrieved_chunks)
        return combined_context

    # Display chat history
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            st.markdown(f"**You:** {entry['content']}")
        else:
            st.markdown(f"**Bot:** {entry['content']}")

    # If the user submits a question
    if user_input:
        # Add the user's question to the chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Retrieve relevant chunks from the dataset
        combined_context = retrieve_chunks(user_input)

        # Generate response based on the retrieved chunks
        response = generate_text_with_hf(combined_context, max_length=max_length)

        # Add bot's response to the chat history
        st.session_state.chat_history.append({"role": "bot", "content": response})

    # Chat input style for a seamless experience
    st.write("---")
    st.text_input("Your next question", key="new_input", placeholder="Ask another question...", on_change=None)

if __name__ == "__main__":
    main()
