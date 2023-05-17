# LangChain Documentation Helper

## Upload langchain docs to a vector index in pinecone - upload.py
1. downloaded the langchain latests docs in repository format
2. used `ReadTheDocsLoader` to parse it to text
3. `RecursiveCharacterTextSplitter` divides the text to chunks of 500 chars 
4. parse the resulting 12100 chunks to vectors using the `OpenAi` embeddings model
5. Finally store the vectors in pinecone `langchain-doc-index`

## Setting up BE - core.py
we want to ask the open ai model: `what is a langchain chain?`
problem, the model is not trained it doesn't know. So we need to give it some context to answer
we have information to add as context in our vector store

How this is solved using `RetrievalQA` chain
1. take the question, parse it to a vector and send it to the vector store
2. the vector store (pinecone) does a similarity search to find a couple of vectors close to the question
3. this relevant vectors will be our context 
4. the chain is smart enough to then take the new context, parse text and plug it with the original question (augment the question)
5. Now we send this full prompt with relevant semantic context to the open ai llm and request to answer the question based on the context that we provide

> :bulb: this is a common pattern or strategy usually referenced as Embedding [See others](https://www.promptengineering.org/master-prompt-engineering-llm-embedding-and-fine-tuning/)
> 