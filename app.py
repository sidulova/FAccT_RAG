from flask import Flask, request, jsonify, render_template
from langchain.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Initialize Flask app
app = Flask(__name__)

# Initialize the QA system
def initialize_qa_system():
    # Load the vector database
    faiss_db = FAISS.load_local("papers_vdb", HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)
    retriever = faiss_db.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 chunks
    llm = OpenAI()

    prompt_template = """Answer the following question based on the context:

    Context: {context}
    Question: {input}

    Answer:"""
    prompt = PromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create retrieval chain
    qa_chain = create_retrieval_chain(retriever, document_chain)
    return qa_chain

# Initialize the QA system
qa_chain = initialize_qa_system()

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# API endpoint for questions
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        query = {"input": data['question']}
        response = qa_chain.invoke(query)

        # Extract source documents
        source_documents = response.get('context', [])
        sources = []

        if source_documents:
            for i, doc in enumerate(source_documents):
                sources.append({
                    "chunk_id": i + 1,
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

        return jsonify({
            "answer": response['answer'],
            "sources": sources
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Direct query endpoint (for testing)
@app.route('/query/<question>')
def query(question):
    try:
        query = {"input": question}
        response = qa_chain.invoke(query)

        # Extract source documents
        source_documents = response.get('context', [])
        sources = []

        if source_documents:
            for i, doc in enumerate(source_documents):
                sources.append({
                    "chunk_id": i + 1,
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

        return jsonify({
            "answer": response['answer'],
            "sources": sources
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add a simple HTML template for the frontend
@app.route('/templates/index.html')
def serve_template():
    return


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
