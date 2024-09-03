import streamlit as st
from ragChain import getRAGChain, getAnswerFromRAGChain
from test import *
import pickle
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
    vector_store = load_vector_store(persist_directory="chroma_store")  # Specify your persist directory
    return vector_store.as_retriever()

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
            rag_chain = initialize_rag_chain(retriever)

            # Generate results for each output type
            answer = getAnswerFromRAGChain(rag_chain, input_str, "answer")
            evidence = getAnswerFromRAGChain(rag_chain, input_str, "evidence")
            auditor_questions = getAnswerFromRAGChain(rag_chain, input_str, "auditor")
            employee = getAnswerFromRAGChain(rag_chain, input_str, "employee")

            # Display the results
            st.write("**Answer:**", answer)
            st.write("**Evidence:**", evidence)
            st.write("**Auditor Questions:**", auditor_questions)
            st.write("**Employee:**", employee)

# Multi-page setup
page_names_to_funcs = {
    "CSV Input": main_page,
    "Manual Input": manual_input_page,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]() # type: ignore
