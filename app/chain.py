from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .prompts import RAG_PROMPT
from .retriever import get_retriever


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


def create_rag_chain():

    retriever = get_retriever()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    def retrieve_and_format(question):

        documents = retriever.invoke(
            question
        )

        return format_documents(
            documents
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




