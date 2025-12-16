import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google as genai
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate


def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF file objects."""
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    """Splits the raw text into smaller chunks."""
    # Note: 10000 chunk size is very large and might exceed token limits for context window.
    # Consider reducing for better performance, e.g., to 1000 or 4000.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, api_key):
    """Creates embeddings and a FAISS vector store."""
    # Pass the API key to the embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain(api_key):
    """Creates the LangChain conversational QA chain."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    # API key to the Chat model
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3,
                                   google_api_key=api_key) # Pass the key here

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question, api_key):
    """Performs similarity search, retrieves context, and gets the final answer."""
    
    # API key to the embeddings model when loading the FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key=api_key)
    
    # Check if the FAISS index exists before loading
    if not os.path.exists("faiss_index"):
        st.error("Vector store not found. Please upload and process the PDF files first.")
        return

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(api_key)

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    # st.write("Raw response for debugging: ", response) # Debugging print removed
    st.write("Reply: ", response["output_text"])


#  Streamlit Application 

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    #  Sidebar 
    with st.sidebar:
        st.title("Menu:")
        
        # 1. API Key Input Section 
        st.subheader("1. Gemini API Key")
        # Use st.session_state to persist the key across reruns
        if "gemini_api_key" not in st.session_state:
            st.session_state.gemini_api_key = ""
            
        api_key_input = st.text_input(
            "Enter your Gemini API Key:", 
            type="password", 
            key="api_key_input",
            value=st.session_state.gemini_api_key
        )
        # Update session state when input changes
        if api_key_input:
            st.session_state.gemini_api_key = api_key_input
            os.environ["GEMINI_API_KEY"] = api_key_input # Set env variable temporarily
            st.success("API Key stored.")
        else:
            st.warning("Please enter your Gemini API Key.")
            
        st.markdown("---")
            
        # 2. PDF Upload Section
        st.subheader("2. Upload & Process PDF")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files:", 
            accept_multiple_files=True
        )
        
        # 3. Process Button
        if st.button("Submit & Process"):
            if not st.session_state.gemini_api_key:
                st.error("Please enter your Gemini API Key first.")
            elif pdf_docs:
                with st.spinner("Processing..."):
                    # Pass the key to the vector store function
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, st.session_state.gemini_api_key)
                    st.success("PDFs Processed and Vector Store Created!")
            else:
                st.warning("Please upload PDF files.")
                
        st.markdown("---")
        st.info("Your API key is only stored in your current browser session and is not saved anywhere else.")

    # --- Main Chat Area ---
    
    # Check if the key is available before accepting user input
    if st.session_state.gemini_api_key:
        user_question = st.text_input("Ask a Question from the Processed PDF Files")

        if user_question:
            # Pass the key to the user_input function
            user_input(user_question, st.session_state.gemini_api_key)
    else:
        st.warning("Please enter your Gemini API Key in the sidebar to enable the chat.")


if __name__ == "__main__":

    main()


