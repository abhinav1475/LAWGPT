import os

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def create_chunks(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for txt_file in os.listdir(input_folder):
        with open(os.path.join(input_folder, txt_file), 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            with open(os.path.join(output_folder, f"{txt_file[:-4]}_chunk_{i}.txt"), 'w', encoding='utf-8') as out_file:
                out_file.write(chunk)

if __name__ == "__main__":
    create_chunks('/content/drive/MyDrive/LawGPT/dataset/chunks', '/content/drive/MyDrive/LawGPT/dataset/chunks')
