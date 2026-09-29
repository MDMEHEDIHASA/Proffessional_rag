from langchain_core.vectorstores import VectorStoreRetriever

from .vectorstore import load_vectorstore


def get_retriever() -> VectorStoreRetriever:

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10, #update for rerankder
            "fetch_k": 20, #update for reranker
            "lambda_mult": 0.7,
        },
    )

    return retriever