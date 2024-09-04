import pandas as pd
import pickle
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
import pdfplumber
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from ragChain import *
from langchain_core.documents.base import Document as BaseDocument
import csv
import pickle
from semantic_test import *
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
import pandas as pd

def getRAGChain_PDF(retriever, llm):
    answer_prompt = ChatPromptTemplate.from_messages(
        [("system", """You are an intelligent assistant with access to a specific PDF document. 
          Your task is to provide accurate, concise, and relevant answers to questions based on the information contained in this PDF. 
          When responding, ensure that you extract and use only the information from the document, without making assumptions or adding outside knowledge. 
          If the question is not addressed in the PDF, state that the answer cannot be found in the document. Be clear and precise in your responses.
          context = {context}"""),
         ("human", "{input}")]
    )
    answer_chain = create_stuff_documents_chain(llm, answer_prompt)
    
    # Create retrieval chains
    chains = {
        'answer': create_retrieval_chain(retriever, answer_chain)
    }
    
    return chains

def getAnswerFromRAGChain_PDF(chains, prompt, output_type):
    if output_type not in chains:
        raise ValueError(f"Invalid output type: {output_type}")
    
    rag_chain = chains[output_type]
    
    try:
        results = rag_chain.invoke({"input": prompt})
        return results.get('answer', '')
    
    except Exception as e:
        print(f"Error processing {output_type}: {e}")
        return ''
    
    

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

def create_PDF_vector_store(documents, persist_directory="pdf_semantic_chroma_store"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    vector_store.persist()
    return vector_store

def load_PDF_vector_store(persist_directory="pdf_semantic_chroma_store"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vector_store


pdf_path = "HR Policy Manual 1.pdf"  # Update with your PDF path
persist_directory = "pdf_semantic_chroma_store"

if not os.path.exists(persist_directory):
        # Load and prepare documents using semantic chunking
    documents = load_and_prepare_documents(pdf_path)
        
        # Create and persist the vector store
    vector_store = create_PDF_vector_store(documents, persist_directory)
else:
        # Load the existing vector store
    vector_store = load_PDF_vector_store(persist_directory)

retriever = vector_store.as_retriever()

llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )#type:ignore

chains = getRAGChain_PDF(retriever, llm)

query = "What is the key topic discussed on the first page of the document?"
output_type = 'answer'
# answer_questions_from_csv(input_csv, output_csv, chains)
answer = getAnswerFromRAGChain_PDF(chains, query, output_type)

# Print the result
print(f"Query: {query}")
print(f"Answer: {answer}")