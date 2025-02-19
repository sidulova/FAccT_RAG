# My personal papers RAG system
RAG System from my personal Scientific PDFs

## Project Summary
- **PDF Parsing**: Extract text from scientific PDFs.
- **Chunking**: Split large documents into meaningful sections.
- **Embedding**: Convert text into vector representations for efficient retrieval.
- **Vector Storage**: Store embeddings in a database (FAISS).
- **Retrieval & LLM Integration**: Retrieve relevant chunks and generate responses.
- **API**: Deploy RAG as a Flask service.


## Installation

```bash
git clone https://github.com/yourusername/rag_from_pdfs.git
cd rag_from_pdfs
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```
