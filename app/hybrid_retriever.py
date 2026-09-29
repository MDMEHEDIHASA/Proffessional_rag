from langchain_community.retrievers import BM25Retriever

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from .vectorstore import load_vectorstore


def create_bm25_retriever(chunks):

    bm25_retriever = BM25Retriever.from_documents(
        chunks,
        k=10,
    )

    return bm25_retriever


def create_hybrid_retriever(chunks):

    # -------------------------
    # 1. BM25
    # -------------------------

    bm25_retriever = create_bm25_retriever(chunks)


    # -------------------------
    # 2. FAISS
    # -------------------------

    vectorstore = load_vectorstore()

    faiss_retriever = vectorstore.as_retriever(
        #search_kwargs={"k": 10}
        search_type="mmr",
        search_kwargs={
            "k": 6, #update for rerankder
            "fetch_k": 12, #update for reranker
            "lambda_mult": 0.7,
            },
    )


    # -------------------------
    # 3. Hybrid
    # -------------------------

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            faiss_retriever,
        ],

        weights=[
            0.5,
            0.5,
        ],
    )


    return hybrid_retriever


