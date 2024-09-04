import streamlit as st
from ragChain import getRAGChain, getAnswerFromRAGChain
import rag_pdf
from test import *
from semantic_test import *
from rag_pdf import *
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# load_dotenv()

# Load API keys from environment variables
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# groq_api_key = os.getenv('GROQ_API_KEY')
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Function to load the saved vector store and initialize the retriever
def initialize_retriever():
    st.write("Loading vector store...")
    vector_store = load_vector_store(persist_directory="token_chroma_store")  # Specify your persist directory
    return vector_store.as_retriever()

def initialize_semantic_retriever():
    st.write("Loading semantic vector store...")
    semantic_vector_store = load_semantic_vector_store(persist_directory="semantic_chroma_store")  # Specify your persist directory
    return semantic_vector_store.as_retriever()    

# Function to initialize the language model and RAG chain
def initialize_rag_chain(retriever):
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    ) # type: ignore
    return getRAGChain(retriever, llm)

def initialize_rag_chain_manual(retriever):
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    ) # type: ignore
    return getRAGChain_Manual(retriever, llm)

# Main page for CSV input
def main_page():
    st.title("CMMI PDF Question Answering App")

    retriever = initialize_retriever()
    rag_chain = initialize_rag_chain(retriever)

    input_csv = st.file_uploader("Upload a CSV file with questions", type="csv")
    if input_csv is not None:
        input_csv_path = "input.csv"  # Temporary path for the uploaded CSV
        with open(input_csv_path, "wb") as f:
            f.write(input_csv.getbuffer())

        output_csv_path = "output.csv"

        st.write("Answering questions...")
        answer_questions_from_csv(input_csv_path, output_csv_path, rag_chain)

        st.success("Questions answered successfully!")
        with open(output_csv_path, "rb") as f:
            st.download_button("Download output CSV", f, file_name="output.csv")

def manual_input_page():
    st.title("Manual Input for CMMI Questions")

    # Combine all inputs into a single string
    st.write("Please enter the following details:")
    practice_area = st.text_input("Practice Area")
    level = st.text_input("Level")
    sub_level = st.text_input("Sub-Level")
    question = st.text_area("Question")
    additional_info = st.text_area("Additional Info")

    # Button to trigger the generation of answers
    if st.button("Get Answers"):
        if not practice_area or not level or not sub_level or not question or not additional_info:
            st.error("Please fill out all fields before generating answers.")
        else:
            # Combine the inputs into a single string formatted for the model
            input_str = (
                f"Practice Area: {practice_area}\n"
                f"Level: {level}\n"
                f"Sub-Level: {sub_level}\n"
                f"Question: {question}\n"
                f"Additional Info: {additional_info}"
            )

            # Initialize retriever and RAG chain only when the button is clicked
            retriever = initialize_retriever()
            rag_chain = initialize_rag_chain_manual(retriever)

            # Generate results for each output type
            answer = getAnswerFromRAGChain_Manual(rag_chain, input_str, "answer")
            evidence = getAnswerFromRAGChain_Manual(rag_chain, input_str, "evidence")
            combined = getAnswerFromRAGChain_Manual(rag_chain, input_str, "combined")
            # auditor_questions = getAnswerFromRAGChain(rag_chain, input_str, "auditor")
            # employee = getAnswerFromRAGChain(rag_chain, input_str, "employee")

            # Display the results
            st.write("**Answer:**", answer)
            st.write("**Evidence:**", evidence)
            st.write("**Q&A:**", combined)
            # st.write("**Auditor Questions:**", auditor_questions)
            # st.write("**Employee:**", employee)

def semantic_main_page():
    st.title("CMMI PDF Question Answering App(Semantic)")

    retriever = initialize_semantic_retriever()
    rag_chain = initialize_rag_chain(retriever)

    input_csv = st.file_uploader("Upload a CSV file with questions", type="csv")
    if input_csv is not None:
        input_csv_path = "input.csv"  # Temporary path for the uploaded CSV
        with open(input_csv_path, "wb") as f:
            f.write(input_csv.getbuffer())

        output_csv_path = "output.csv"

        st.write("Answering questions...")
        answer_questions_from_csv(input_csv_path, output_csv_path, rag_chain)

        st.success("Questions answered successfully!")
        with open(output_csv_path, "rb") as f:
            st.download_button("Download output CSV", f, file_name="output.csv")

def semantic_manual_input_page():
    st.title("Manual Input for CMMI Questions(Semantic)")

    # Combine all inputs into a single string
    st.write("Please enter the following details:")
    practice_area = st.text_input("Practice Area")
    level = st.text_input("Level")
    sub_level = st.text_input("Sub-Level")
    question = st.text_area("Question")
    additional_info = st.text_area("Additional Info")

    # Button to trigger the generation of answers
    if st.button("Get Answers"):
        if not practice_area or not level or not sub_level or not question:
            st.error("Please fill out all fields before generating answers.")
        else:
            # Combine the inputs into a single string formatted for the model
            input_str = (
                f"Practice Area: {practice_area}\n"
                f"Level: {level}\n"
                f"Sub-Level: {sub_level}\n"
                f"Question: {question}\n"
                f"Additional Info: {additional_info}"
            )

            # Initialize retriever and RAG chain only when the button is clicked
            retriever = initialize_semantic_retriever()
            rag_chain = initialize_rag_chain_manual(retriever)

            # Generate results for each output type
            answer = getAnswerFromRAGChain_Manual(rag_chain, input_str, "answer")
            evidence = getAnswerFromRAGChain_Manual(rag_chain, input_str, "evidence")
            combined = getAnswerFromRAGChain_Manual(rag_chain, input_str, "combined")
            # auditor_questions = getAnswerFromRAGChain_Manual(rag_chain, input_str, "auditor")
            # employee = getAnswerFromRAGChain_Manual(rag_chain, input_str, "employee")

            # Display the results
            st.write("**Answer:**", answer)
            st.write("**Evidence:**", evidence)
            st.write("**Q&A:**", combined)
            # st.write("**Auditor Questions:**", auditor_questions)
            # st.write("**Employee:**", employee)
            
def PDF_QA():
    st.title("PDF Q&A with RAG-LLM")
    st.write("Upload a PDF, ask a question, and get an answer based on the content of the PDF.")

    # Upload PDF file
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # Ask a question
    question = st.text_input("Ask a question about the PDF:")

    if uploaded_file and question:
        # Save uploaded file temporarily
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Define persistence directory for the vector store
        persist_directory = "pdf_semantic_chroma_store"

        # Check if vector store already exists
        if not os.path.exists(persist_directory):
            # Load and prepare documents using semantic chunking
            documents = load_and_prepare_documents(pdf_path)
            
            # Create and persist the vector store
            vector_store = create_PDF_vector_store(documents, persist_directory)
        else:
            # Load the existing vector store
            vector_store = load_PDF_vector_store(persist_directory)

        # Create the retriever
        retriever = vector_store.as_retriever()

        # Initialize the LLM
        llm = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        ) #type:ignore

        # Get RAG Chain
        chains = getRAGChain_PDF(retriever, llm)

        # Get answer from RAG chain
        output_type = 'answer'
        answer = getAnswerFromRAGChain_PDF(chains, question, output_type)

        # Display the answer
        st.write(f"**Question:** {question}")
        st.write(f"**Answer:** {answer}")




# Multi-page setup
page_names_to_funcs = {
    "CSV Input": main_page,
    "Manual Input": manual_input_page,
    "Semantic CSV Input": semantic_main_page,
    "Semantic Manual Input": semantic_manual_input_page,
    "PDF QA": PDF_QA
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]() # type: ignore
