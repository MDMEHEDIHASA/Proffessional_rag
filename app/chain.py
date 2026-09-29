from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .prompts import RAG_PROMPT
from .retriever import get_retriever
from .hybrid_retriever import create_hybrid_retriever
from .reranker import DocumentReranker


def format_documents(
    documents: list[Document],
) -> str:

    formatted = []

    for doc in documents:

        source = doc.metadata.get(
            "source",
            "unknown",
        )

        page = doc.metadata.get(
            "page",
            "unknown",
        )

        formatted.append(
            f"""
Source: {source}
Page: {page}

{doc.page_content}
"""
        )

    return "\n\n---\n\n".join(formatted)


def create_rag_chain(chunks):

    #retriever = get_retriever() //comment out for using hybrid retriever
    # Usse Hybrid Retriever
    #Step 1 Hybrid encoder
    retriever = create_hybrid_retriever(chunks)

    # Step 2: Cross-Encoder Reranker 
    reranker = DocumentReranker( model_name="BAAI/bge-reranker-base" )

    # Step 3: LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    # Step 4: Retrieve → Rerank → Format
    def retrieve_and_format(question):

        # Retrieve candidate documents using Hybrid Retriever
        ## hybrid documents
        candidates = retriever.invoke(
            question
        )

        # Rerank candidates using Cross-Encoder

        final_documents = reranker.rerank( 
            query=question, 
            documents=candidates, 
            top_k=4, 
        )


        return format_documents(
            final_documents
        )

    chain = (
        {
            "context": retrieve_and_format,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain




