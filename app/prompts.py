from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Answer the user's question using ONLY the
provided context.

If the answer cannot be found in the context,
say that you do not have enough information.

Do not invent facts.

Always cite the source documents when possible.
"""
        ),

        (
            "human",
            """
Context:

{context}

Question:

{question}
"""
        ),
    ]
)