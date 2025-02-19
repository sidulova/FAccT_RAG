from langchain.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

faiss_db = FAISS.load_local("papers_vdb", HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)
retriever = faiss_db.as_retriever()
llm = OpenAI()

prompt_template = """Answer the following question based on the context:

Context: {context}
Question: {input}

Answer:"""
prompt = PromptTemplate.from_template(prompt_template)
document_chain = create_stuff_documents_chain(llm, prompt)


from langchain.chains import create_retrieval_chain
qa_chain = create_retrieval_chain(retriever, document_chain)
query = {"input": "What is bias in medicine?"}
response = qa_chain.invoke(query)
print(response['answer'])



