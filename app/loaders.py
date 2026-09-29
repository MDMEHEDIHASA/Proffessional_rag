from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_documents(directory: str) -> list[Document]:

    documents = []

    pdf_files = Path(directory).glob("*.pdf")

    for pdf_file in pdf_files:

        loader = PyPDFLoader(str(pdf_file))

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = pdf_file.name

        documents.extend(docs)

    return documents