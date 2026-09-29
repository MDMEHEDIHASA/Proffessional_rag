from sentence_transformers import CrossEncoder

class DocumentReranker:

    def __init__(
        self,
        model_name="BAAI/bge-reranker-base",
    ):

        self.model = CrossEncoder(
            model_name
        )


    def rerank(
        self,
        query,
        documents,
        top_k=4,
    ):

        pairs = [
            [query, doc.page_content]
            for doc in documents
        ]

        scores = self.model.predict(
            pairs
        )

        ranked = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            doc
            for score, doc in ranked[:top_k]
        ]