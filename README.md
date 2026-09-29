                    ┌──────────────────┐
                    │   PDF Documents  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │    PDF Loader    │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Text Chunking    │
                    │ Semantic/Recursive│
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │    Embeddings    │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Vector Database  │
                    │      FAISS       │
                    └────────┬─────────┘
                             │
                  User Question
                             ↓
                    ┌──────────────────┐
                    │ Query Embedding  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ MMR Retrieval    │
                    │ relevant+diverse │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Context + Query  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │      Prompt      │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │       LLM        │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Final Answer +   │
                    │ Source Documents │
                    └──────────────────┘



### 1. Project structure
```
professional-rag/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── loaders.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── prompts.py
│   ├── chain.py
│   └── main.py
│
├── data/
│   └── documents/
│
├── vectorstore/
│
├── .env
├── .gitignore
├── pyproject.toml
└── README.md
```

## 2. Environment setup
mkdir professional-rag
cd professional-rag

uv init

uv python install 3.12

uv venv --python 3.12

source .venv/bin/activate


## Next steps
uv add langchain
uv add langchain-community
uv add langchain-openai
uv add langchain-text-splitters
uv add langchain-experimental
uv add faiss-cpu
uv add pypdf
uv add python-dotenv


## 3. .env
OPENAI_API_KEY=your_api_key_here



###  Ulimate goal learning
```
1. Document ingestion
2. Recursive chunking
3. Semantic chunking
4. Embedding models
5. Vector databases
6. Similarity search
7. MMR
8. Hybrid search
9. Reranking
10. Metadata filtering
11. Query transformation
12. Multi-query retrieval
13. Conversational RAG
14. Agents
15. Tool calling
16. Structured output
17. Evaluation
18. Tracing/observability
19. FastAPI
20. Docker
21. AWS/GCP deployment
22. Security
23. Latency optimization
24. Cost optimization
```
