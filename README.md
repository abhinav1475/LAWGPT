# LawGPT

LawGPT is a question-answering system that leverages Hugging Face's Language Models and FAISS for efficient retrieval and answering of legal queries. This project processes legal documents in PDF format, chunks the text, and uses vector embeddings for fast and accurate retrieval of relevant information.

## Setup

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Place your legal PDF documents in the `dataset/pdfs/` folder.

3. Run the project:
    ```bash
    python src/main.py
    ```

## Components

- `pdf_processing.py`: Extracts text from PDFs.
- `chunking.py`: Splits extracted text into manageable chunks.
- `vector_store.py`: Converts chunks into vector embeddings and stores them in FAISS.
- `retriever.py`: Retrieves the most relevant chunks using FAISS.
- `qa_pipeline.py`: Uses a Hugging Face model to answer questions based on retrieved text.

## Example Usage

Ask a question:
```bash
python src/main.py
