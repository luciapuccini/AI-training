import os
from langchain.document_loaders import ReadTheDocsLoader

# This text splitter is the recommended one for generic text.
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

from constants import PINECONE_INDEX_NAME

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(
        path="langchain-docs/langchain-docs/python.langchain.com/en/latest",
        encoding="utf-8",
    )
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    # prompt = query/question + context  < openai llm TOKEN limit ~4000
    # ie reserving 2000 tokens for context, which translates to chunk size
    # we DO NOT want to make the chuck size too small or it will lose semantic meaning
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # ex chunk size of 400 --> 15443 chunks. size 500 --> 12100 chunks
    chunks = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(chunks)} chunks")

    for text in chunks:
        old_path = text.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        text.metadata.update({"source": new_url})

    embeddingLLM = OpenAIEmbeddings()
    # persist 12100 chunks parsed to vectors (12100 vectors in pinecone)
    Pinecone.from_documents(chunks, embeddingLLM, index_name=PINECONE_INDEX_NAME)


if __name__ == "__main__":
    ingest_docs()
