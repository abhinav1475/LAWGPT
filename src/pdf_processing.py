import fitz  # PyMuPDF
import os

def extract_text_from_pdfs(pdf_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            doc = fitz.open(os.path.join(pdf_folder, pdf_file))
            text = ""
            for page in doc:
                text += page.get_text()
            with open(os.path.join(output_folder, f"{pdf_file}.txt"), 'w', encoding='utf-8') as f:
                f.write(text)

if __name__ == "__main__":
    extract_text_from_pdfs('/content/drive/MyDrive/LawGPT/dataset/pdfs', '/content/drive/MyDrive/LawGPT/dataset/chunks')
