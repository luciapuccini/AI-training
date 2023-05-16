import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import  FAISS

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if __name__ == "__main__":
    loader = PyPDFLoader(
        "C:/Users/pucci/Desktop/Dev/ai-training/vectorstore-in-memory/transformers.pdf"
    )
    pages = loader.load()


    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )

    docs = text_splitter.split_documents(documents=pages)
    print(docs)

    #emmbedings model from open AI - no data here
    embeddings = OpenAIEmbeddings()
    # takes data & embedding model -> vector representation
    vectorstore = FAISS.from_documents(docs,embeddings)
    # save the vectors into an index
    vectorstore.save_local('faiss_index_transformers_paper')