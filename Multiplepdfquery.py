import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Set Google API Key
GOOGLE_API_KEY = 'AIzaSyDnrbE2jYw9wGrkrEHX2ic4MrU1E9XU_lU'
genai.configure(api_key=GOOGLE_API_KEY)


def main():
    st.title("PDF Question Answering with Google Generative AI")
    st.sidebar.title("Upload PDFs")
    pdf_files = st.sidebar.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

    if pdf_files:
        text = ""
        for pdf in pdf_files:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        text_chunks = text_splitter.split_text(text)

        # Create and save vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

        # Create conversational chain
        prompt_template = """
        answer the question as detail as possible from the provided context if there is no answer in the context then just say "answer is not available in the context",don't provide the
        context:\n {context}?\n
        Question:\n {question}\n
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
        prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        # User input loop
        user_question = st.text_input("Ask a question from the PDF Files", "")
        if user_question:
            docs = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model='models/embedding-001',
                                                                                google_api_key=GOOGLE_API_KEY)).similarity_search(
                user_question)

            # Error handling for cases where the model is unable to find an answer
            if not docs:
                st.write("Unable to find relevant context in the PDF files.")
            else:
                response = chain({'input_documents': docs, 'question': user_question}, return_only_outputs=True)
                if response == "answer is not available in the context":
                    st.write("Answer is not available in the provided context.")
                else:
                    st.write("Response:", response)


if __name__ == "__main__":
    main()
