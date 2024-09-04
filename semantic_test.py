import os
import pdfplumber
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from ragChain import *
from langchain_core.documents.base import Document as BaseDocument
import csv
import pickle
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

class Document(BaseDocument):
    def __init__(self, text):
        super().__init__(page_content=text, metadata={})

def extract_tables_from_pdf(pdf_path):
    tables_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                table_text = '\n'.join(['\t'.join(cell if cell is not None else '' for cell in row) for row in table])
                tables_text.append(table_text)
    return tables_text

def process_pdf_document(file_path):
    # Load the PDF document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Number of pages loaded: {len(documents)}")
    
    # Initialize the embedding model
    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create semantic chunks from the document content
    semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")
    print("Semantic chunker initialized:", semantic_chunker)
    semantic_chunks = semantic_chunker.create_documents([d.page_content for d in documents])
    print("Number of semantic chunks created:", len(semantic_chunks))
    
    return semantic_chunks

def load_and_prepare_documents(pdf_path):
    # Extract tables as text from the PDF
    tables_text = extract_tables_from_pdf(pdf_path)
    
    # Process the PDF to create semantic chunks
    semantic_chunks = process_pdf_document(pdf_path)
    
    # Combine semantic chunks with table texts
    combined_texts = semantic_chunks + [Document(text) for text in tables_text]
    
    return combined_texts

def create_semantic_vector_store(documents, persist_directory="semantic_chroma_store"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    vector_store.persist()
    return vector_store

def load_semantic_vector_store(persist_directory="semantic_chroma_store"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vector_store

def answer_questions_from_csv(input_csv, output_csv, chains):
    df = pd.read_csv(input_csv)
    
    required_columns = ['Practice Area', 'Level', 'Sub-Level', 'Question', 'Additional Info']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Input CSV must contain the '{col}' column.")
    
    answers, evidences, auditors, employees = [], [], [], []
    
    for _, row in df.iterrows():
        prompt = (
            f"Practice Area: {row.get('Practice Area', '')}\n"
            f"Level: {row.get('Level', '')}\n"
            f"Sub-Level: {row.get('Sub-Level', '')}\n"
            f"Question: {row.get('Question', '')}\n"
            f"Additional Info: {row.get('Additional Info', '')}"
        )
        
        answer = getAnswerFromRAGChain(chains, prompt, 'answer')
        evidence = getAnswerFromRAGChain(chains, prompt, 'evidence')
        auditor = getAnswerFromRAGChain(chains, prompt, 'auditor')
        answers.append(answer)
        evidences.append(evidence)
        auditors.append(auditor)

    df['Answer'] = answers
    df['Evidence'] = evidences
    df['Auditor Questions'] = auditors

    for _, row in df.iterrows():
        employee_prompt = (
            f"Practice Area: {row.get('Practice Area', '')}\n"
            f"Level: {row.get('Level', '')}\n"
            f"Sub-Level: {row.get('Sub-Level', '')}\n"
            f"Question: {row.get('Question', '')}\n"
            f"Additional Info: {row.get('Additional Info', '')}\n"
            f"Please provide a clear and accurate response to each and every question asked: {row.get('Auditor Questions', '')}"
        )
        employee = getAnswerFromRAGChain(chains, employee_prompt, 'employee')
        employees.append(employee)

    df['Employee'] = employees
    
    try:
        df.to_csv(output_csv, index=False)
        print(f"Output successfully written to {output_csv}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")

def main():
    pdf_path = "CMMI.pdf"  # Update with your PDF path
    persist_directory = "semantic_chroma_store"

    if not os.path.exists(persist_directory):
        # Load and prepare documents using semantic chunking
        documents = load_and_prepare_documents(pdf_path)
        
        # Create and persist the vector store
        vector_store = create_semantic_vector_store(documents, persist_directory)
    else:
        # Load the existing vector store
        vector_store = load_semantic_vector_store(persist_directory)

    retriever = vector_store.as_retriever()

    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )#type:ignore

    chains = getRAGChain(retriever, llm)

    input_csv = "test_question.csv"
    output_csv = "output.csv"

    answer_questions_from_csv(input_csv, output_csv, chains)

if __name__ == "__main__":
    main()
