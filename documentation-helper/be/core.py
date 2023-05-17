import os
from typing import TypedDict

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone

from constants import PINECONE_INDEX_NAME

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


# connect to the pinecone index - init

# take a promt = question

# parse question to embbeding. maybe we need to split it first

# use the chat model to as a question with this embedding
# answer = QAchain ( ll openAiChat, question )
# ! problem open AI does not know what is langchain ... not trained
#  we need to pass context. prompt = queston + context

class chatResponseType(TypedDict):
    query: str
    result: str
    source_documents: str

def run_llm(query: str) -> chatResponseType:
    embeddings = OpenAIEmbeddings()
    semanticsearch = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    # retriever is a wrapper around the vectorstore which does the similarity search
    qachain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=semanticsearch.as_retriever(),
        return_source_documents=True,
    )
    # run the chain with the passing question
    answer = qachain(query)

    return answer


if __name__ == "__main__":
    print(run_llm(query="What is langchain?"))
