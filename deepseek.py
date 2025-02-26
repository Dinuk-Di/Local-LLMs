import streamlit as st
import time
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# Initialize embeddings and text splitter
def initialize_components():
    embeddings = HuggingFaceEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return embeddings, text_splitter

# RAG pipeline
def rag_pipeline(document, question, model_name, embeddings, text_splitter):
    start_time = time.time()

    # Step 1: Split document into chunks
    with st.spinner("Splitting document into chunks..."):
        chunks = text_splitter.split_text(document)
    st.success("Document split into chunks!")

    # Step 2: Create vector store for retrieval
    with st.spinner("Creating vector store..."):
        vector_store = FAISS.from_texts(chunks, embeddings)
    st.success("Vector store created!")

    # Step 3: Retrieve relevant chunks based on the query
    with st.spinner("Retrieving relevant context..."):
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    st.success("Relevant context retrieved!")

    # Step 4: Generate answer using retrieved context
    with st.spinner("Generating answer using RAG..."):
        llm = Ollama(model=model_name)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        answer = qa_chain.run(question)
    st.success("Answer generated using RAG!")

    end_time = time.time()
    return answer, end_time - start_time

# CAG pipeline
def cag_pipeline(document, question, model_name):
    start_time = time.time()

    # Step 1: Preload the entire document into the LLM context
    with st.spinner("Preloading document into LLM context..."):
        llm = Ollama(model=model_name)
        prompt = f"Document:\n{document}\n\nQuestion:\n{question}\nAnswer:"
    st.success("Document preloaded into LLM context!")

    # Step 2: Generate answer directly from the full document context
    with st.spinner("Generating answer using CAG..."):
        answer = llm(prompt)
    st.success("Answer generated using CAG!")

    end_time = time.time()
    return answer, end_time - start_time

# Streamlit UI for comparison
st.title("RAG vs CAG System Comparison with LangChain and Ollama")
st.write("Upload a document and ask a question to compare Retrieval-Augmented Generation (RAG) and Context-Augmented Generation (CAG) systems.")

# File upload and question input
uploaded_file = st.file_uploader("Upload a document (TXT format)", type=["txt"])
question = st.text_input("Enter your question:")
model_choice = st.selectbox(
    "Select the model to use:",
    ["llama-1b", "deepseek-r1:1.5b"]
)

if uploaded_file and question:
    document = uploaded_file.read().decode("utf-8")
    
    # Initialize components for RAG
    embeddings, text_splitter = initialize_components()

    col1, col2 = st.columns(2)

    # RAG System Execution
    with col1:
        st.subheader("RAG System")
        rag_answer, rag_time = rag_pipeline(document, question, model_choice, embeddings, text_splitter)
        st.markdown(f"**Answer:**\n{rag_answer}")
        st.metric(label="Processing Time", value=f"{rag_time:.2f} seconds")

    # CAG System Execution
    with col2:
        st.subheader("CAG System")
        cag_answer, cag_time = cag_pipeline(document, question, model_choice)
        st.markdown(f"**Answer:**\n{cag_answer}")
        st.metric(label="Processing Time", value=f"{cag_time:.2f} seconds")

    # Comparison Table
    st.divider()
    st.subheader("System Comparison")
    
    comparison_data = {
        "Metric": ["Processing Time", "Context Handling", "Answer Quality"],
        "RAG": [
            f"{rag_time:.2f}s", 
            "Dynamic Retrieval (relevant chunks)", 
            "Focused on retrieved sections"
        ],
        "CAG": [
            f"{cag_time:.2f}s", 
            "Entire Document Preloaded", 
            "Holistic understanding of the document"
        ]
    }
    
    st.table(comparison_data)
