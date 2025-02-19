import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""

def chunk_text(text, metadata, chunk_size=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text], [metadata])

# Process all PDFs in the papers folder
papers_dir = "./papers"
all_documents = []

for filename in os.listdir(papers_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(papers_dir, filename)
        print(f"Processing: {pdf_path}")
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        
        if pdf_text:
            # Create metadata to track source file
            metadata = {"source": filename}
            
            # Chunk the text and add to our collection
            chunks = chunk_text(pdf_text, metadata)
            all_documents.extend(chunks)

print(f"Created {len(all_documents)} chunks from {len([f for f in os.listdir(papers_dir) if f.endswith('.pdf')])} papers")

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings()

# Create FAISS index from all documents
if all_documents:
    faiss_db = FAISS.from_documents(all_documents, embeddings)
    faiss_db.save_local("papers_vdb")
    print(f"FAISS index created and saved with {len(all_documents)} chunks")
else:
    print("No documents were processed. Check if there are PDF files in the 'papers' directory.")
