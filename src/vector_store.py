import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def create_vector_store(chunk_folder, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    vectors = []
    chunk_paths = []
    
    for chunk_file in os.listdir(chunk_folder):
        with open(os.path.join(chunk_folder, chunk_file), 'r', encoding='utf-8') as f:
            chunk_text = f.read()
            vector = model.encode([chunk_text])[0]
            vectors.append(vector)
            chunk_paths.append(os.path.join(chunk_folder, chunk_file))
    
    vectors = np.array(vectors)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    
    faiss.write_index(index, 'vector_store.index')
    with open('chunk_paths.txt', 'w') as f:
        f.write('\n'.join(chunk_paths))

if __name__ == "__main__":
    create_vector_store('/content/drive/MyDrive/LawGPT/dataset/chunks')
