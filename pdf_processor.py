from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    
    reader = PdfReader(pdf_path)
    
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
        
    return full_text

def chunk_text(text , chunk_size = 1000 , overlap = 200):
    chunks = []
    start = 0 
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start = end - overlap

    return chunks
    