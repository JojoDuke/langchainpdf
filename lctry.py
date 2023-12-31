## MODULES
import streamlit as st
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import os

with st.sidebar:
    st.title('Chat with PDF AI app - dev')
    
# Load dotenv file
load_dotenv()

def main():
    st.write("Chat with a PDF or CSV")
    
    # Section to upload pdf file
    pdf = st.file_uploader("Upload a PDF", type='pdf')
    csv = st.file_uploader("Upload a CSV", type='csv')
    
    if pdf is not None:
        pdfReader = PdfReader(pdf)
        
        # Split the text in the document up
        text = ""
        for page in pdfReader.pages:
            text += page.extract_text()
            
        textSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = textSplitter.split_text(text=text)
        
        # Compute text embeddings
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            vectorStore = FAISS.from_texts(chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorStore, f)
                
                
        # Textbox to accept user queries
        query = st.text_input("Ask your PDF anything:")
        st.write(query)
        
        if query:
            # Do semantic search
            docs = vectorStore.similarity_search(query=query, k=3)
            
            llm = ChatOpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            
            st.write(response)
            
        
        
        
        

if __name__ == '__main__':
    main()
