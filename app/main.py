from .loaders import load_documents
from .chunking import split_documents
from .vectorstore import create_vectorstore
from .chain import create_rag_chain

from dotenv import load_dotenv

load_dotenv()


def build_knowledge_base():

    print("Loading documents...")

    documents = load_documents(
        "./data/documents"
    )

    print(
        f"Loaded {len(documents)} documents"
    )

    print("Splitting documents...")

    chunks = split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks"
    )

    print("Creating vector store...")

    create_vectorstore(
        chunks
    )

    print("Vector store created!")


def ask_question():

    rag_chain = create_rag_chain()

    while True:

        question = input(
            "\nAsk a question "
            "(type 'exit' to quit): "
        )

        if question.lower() == "exit":
            break

        answer = rag_chain.invoke(
            question
        )

        print("\nAI:")
        print(answer)


if __name__ == "__main__":

    build_knowledge_base()

    ask_question()