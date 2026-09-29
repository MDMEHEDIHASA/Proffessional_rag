from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .embeddings import get_embeddings


VECTORSTORE_PATH = "vectorstore"


def create_vectorstore(
    chunks: list[Document],
):

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
    )

    Path(VECTORSTORE_PATH).mkdir(
        exist_ok=True
    )

    vectorstore.save_local(
        VECTORSTORE_PATH
    )

    return vectorstore


def load_vectorstore():

    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore