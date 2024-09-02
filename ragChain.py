import pandas as pd
import pickle
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Define the RAG Chain
def getRAGChain(retriever, llm):
    answer_prompt = ChatPromptTemplate.from_messages(
        [("system", """You are a Capability Maturity Model Integration (CMMI) expert with deep knowledge of the framework and its application. Your task is to analyze the following query based on the CMMI information provided in the PDF document. The query includes:
                     1. **Practice Area:** The specific domain within the CMMI framework.
                     2. **Level:** The CMMI maturity level relevant to the query.
                     3. **Sub-Level:** The specific aspect or sub-category of the practice area.
                     4. **Question:** The primary query requiring analysis.
                     5. **Additional Info:** Supplementary details that may aid in understanding the query.

                     **Instructions:**

                     1. Carefully interpret the query by identifying the practice area, level, and sub-level.
                     2. Leverage the additional info as context to frame your response.

                     **Output Requirements:**
                     - **Answer:** Provide a precise, well-structured, detailed, and accurate response that directly addresses the question in 3 to 5 bullet points. Clearly define the expectations that should be met according to the CMMI framework. These should be precise, actionable, and directly relevant to the query, reflecting the standards and requirements for the given practice area, level, and sub-level.
                     Ensure your answer is informed by the context and is both unique and reflective of the CMMI principles.
                    Your final output should be comprehensive yet concise, ensuring that each component is uniquely tailored to the query while remaining accurate and aligned with the CMMI framework. Do not write the Query in your output. 
                    context = {context}"""),
         ("human", "{input}")]
    )

    evidence_prompt = ChatPromptTemplate.from_messages(
        [("system", """You are a Capability Maturity Model Integration (CMMI) expert with deep knowledge of the framework and its application. Your task is to analyze the following query based on the CMMI information provided in the PDF document. The query includes:
                     1. **Practice Area:** The specific domain within the CMMI framework.
                     2. **Level:** The CMMI maturity level relevant to the query.
                     3. **Sub-Level:** The specific aspect or sub-category of the practice area.
                     4. **Question:** The primary query requiring analysis.
                     5. **Additional Info:** Supplementary details that may aid in understanding the query.

                     **Instructions:**

                     1. Carefully interpret the query by identifying the practice area, level, and sub-level.
                     2. Leverage the additional info as context to frame your response.

                     **Output Requirements:**
                     - **Evidence:** Cite specific references or details from the CMMI document that substantiate your answer. This should be clear, concise, and directly relevant to the query.
                    Your final output should be comprehensive yet concise, ensuring that each component is uniquely tailored to the query while remaining accurate and aligned with the CMMI framework. Do not include the Query and the contents of the prompt in your output. 
                    context = {context}"""),
         ("human", "{input}")]
    )

    auditor_prompt = ChatPromptTemplate.from_messages(
        [("system", """You are a Capability Maturity Model Integration (CMMI) expert with deep knowledge of the framework and its application. Your task is to analyze the following query based on the CMMI information provided in the PDF document. The query includes:
                     1. **Practice Area:** The specific domain within the CMMI framework.
                     2. **Level:** The CMMI maturity level relevant to the query.
                     3. **Sub-Level:** The specific aspect or sub-category of the practice area.
                     4. **Question:** The primary query requiring analysis.
                     5. **Additional Info:** Supplementary details that may aid in understanding the query.

                     **Instructions:**

                     1. Carefully interpret the query by identifying the practice area, level, and sub-level.
                     2. Leverage the additional info as context to frame your response.

                     **Output Requirements:**
                     - **Auditor:** Formulate a set of around 3 to 5 probing questions that an auditor might ask employees based on the query, aiming to assess compliance or understanding.
                    Your final output should be comprehensive yet concise, ensuring that each component is uniquely tailored to the query while remaining accurate and aligned with the CMMI framework.Do not write the Query in your output. 
                    context = {context}"""),
         ("human", "{input}")]
    )

    employee_prompt = ChatPromptTemplate.from_messages(
        [("system", """You are a Capability Maturity Model Integration (CMMI) expert with deep knowledge of the framework and its application. Your task is to come-up with what the team needs to answer and show evidences to the specific auditor questions and in general what the employee needs to prepare for passing the audit.
                    the following query based on the CMMI information provided in the PDF document. The query includes:
                     1. **Practice Area:** The specific domain within the CMMI framework.
                     2. **Level:** The CMMI maturity level relevant to the query.
                     3. **Sub-Level:** The specific aspect or sub-category of the practice area.
                     4. **Question:** The primary query requiring analysis.
                     5. **Additional Info:** Supplementary details that may aid in understanding the query.
                     6. **Auditor question:** A set of questions asked by the auditor that you are expected to answer.
                     All these points are provided as additional informaiton
                     
                     **Instructions:**

                     1. Carefully interpret the auditor questions and provide responses including implicit expectations.
                    

                     **Output Requirements:**
                     - **Employee:** Offer a guide on how employees should respond to the auditor's questions, ensuring that the Expectations align with CMMI expectations and are framed correctly.
                     List the evidences that the employee is expected to show. 
                    Your final output should be comprehensive yet concise, ensuring that each component is uniquely tailored to the query while remaining accurate and aligned with the CMMI framework. Do not write the Query in your output.
                    context = {context}"""),
         ("human", "The auditor has asked the following questions: {input}. Please provide a clear and accurate response.")]
    )
    
    # Create chains for each output type
    answer_chain = create_stuff_documents_chain(llm, answer_prompt)
    evidence_chain = create_stuff_documents_chain(llm, evidence_prompt)
    auditor_chain = create_stuff_documents_chain(llm, auditor_prompt)
    employee_chain = create_stuff_documents_chain(llm, employee_prompt)
    
    # Create retrieval chains
    chains = {
        'answer': create_retrieval_chain(retriever, answer_chain),
        'evidence': create_retrieval_chain(retriever, evidence_chain),
        'auditor': create_retrieval_chain(retriever, auditor_chain),
        'employee': create_retrieval_chain(retriever, employee_chain),
    }
    
    return chains

def getAnswerFromRAGChain(chains, prompt, output_type):
    if output_type not in chains:
        raise ValueError(f"Invalid output type: {output_type}")
    
    rag_chain = chains[output_type]
    
    try:
        results = rag_chain.invoke({"input": prompt})
        return results.get('answer', '')
    
    except Exception as e:
        print(f"Error processing {output_type}: {e}")
        return ''

    
    

