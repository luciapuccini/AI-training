import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT_REGION = os.environ.get('PINECONE_ENVIRONMENT_REGION')

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT_REGION
)

if __name__ == '__main__':
    loader = TextLoader("C:/Users/pucci/Desktop/Dev/ai-training/intro-to-vector-db/mediumblogs/mediumblog1.txt",
                        'utf-8')
    document = loader.load()

    text_spliter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_spliter.split_documents(document)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    # fist load some data into the vectorstore
    # docstosearch = Pinecone.from_documents(texts, embeddings, index_name='langchain-doc-index')
    # if you already have a populated vectorstore, you can load it like this
    docsearch = Pinecone.from_existing_index(index_name='langchain-doc-index', embedding=embeddings)

    query = "What is a high dimensional space? explained in 10 words or less"
    docs = docsearch.similarity_search(query)
    print(docs[0].page_content)
