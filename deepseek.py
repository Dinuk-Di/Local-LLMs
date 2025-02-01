import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever  # Correct import for BaseRetriever

# Streamlit UI
st.title("📄 RAG System with DeepSeek and Llama Models")

# File uploader for PDF input
uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")

# Initialize session state to preserve data across interactions
if "documents" not in st.session_state:
    st.session_state.documents = None  # Store processed document chunks
if "retriever" not in st.session_state:
    st.session_state.retriever = None  # Store retriever for embedding search

if uploaded_file:
    # Step 1: Save the uploaded file locally with a progress bar
    with st.spinner("Saving uploaded file..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success("File saved successfully!")

    # Step 2: Load and process the PDF document (only if not already processed)
    if st.session_state.documents is None:
        with st.spinner("Loading and processing the PDF document..."):
            loader = PDFPlumberLoader("temp.pdf")
            docs = loader.load()
            st.success("PDF loaded successfully!")

        # Step 3: Split documents into chunks using semantic chunking
        with st.spinner("Splitting document into chunks..."):
            text_splitter = SemanticChunker(HuggingFaceEmbeddings())
            st.session_state.documents = text_splitter.split_documents(docs)
            st.success(f"Document split into {len(st.session_state.documents)} chunks!")

        # Step 4: Create embeddings and vector store for retrieval (only once)
        with st.spinner("Creating embeddings and vector store..."):
            embedder = HuggingFaceEmbeddings()
            vector = FAISS.from_documents(st.session_state.documents, embedder)
            st.session_state.retriever = vector.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            )
            if not isinstance(st.session_state.retriever, BaseRetriever):
                raise ValueError("Failed to initialize a valid retriever.")
            st.success("Embeddings and vector store created successfully!")

# Step 5: Model selection dropdown
model_choice = st.selectbox(
    "Select the model to use:",
    options=["deepseek-r1:1.5b", "llama3.2:1b"],
    index=0  # Default to the first option
)

# Step 6: Initialize the selected model and set up the QA system (only when model changes)
if "current_model" not in st.session_state or st.session_state.current_model != model_choice:
    if not isinstance(st.session_state.retriever, BaseRetriever):
        st.error("Retriever is not properly initialized. Please upload a valid document.")
    else:
        with st.spinner(f"Initializing the {model_choice} model..."):
            llm = Ollama(model=model_choice)

            # Define the QA prompt template
            prompt = """
            Use the following context to answer the question.
            Context: {context}
            Question: {question}
            Answer:"""
            
            QA_PROMPT = PromptTemplate.from_template(prompt)

            llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
            combine_documents_chain = StuffDocumentsChain(
                llm_chain=llm_chain, document_variable_name="context"
            )
            
            # Update session state for QA system and current model
            st.session_state.qa = RetrievalQA(
                combine_documents_chain=combine_documents_chain,
                retriever=st.session_state.retriever,
            )
            st.session_state.current_model = model_choice
            st.success(f"{model_choice} model initialized successfully!")

# Step 7: User input for asking questions about the document
user_input = st.text_input("Ask a question about your document:")

if user_input:
    if "qa" not in st.session_state or not isinstance(st.session_state.qa, RetrievalQA):
        st.error("QA system is not properly initialized. Please check your setup.")
    else:
        with st.spinner("Processing your question..."):
            response = st.session_state.qa(user_input)["result"]
            st.success("Question processed successfully!")
            st.write("**Response:**")
            st.write(response)
