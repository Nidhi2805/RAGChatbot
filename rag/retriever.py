import os
import pickle
import re

from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CORPUS_PATH = "vectorstore_corpus.pkl"

# Lazy globals — nothing loads at import time
_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings


# CrossEncoder removed — replaced with BM25 score fusion reranking
# Saves ~300MB RAM on free tier
def rerank_with_bm25_fusion(query: str, docs: list) -> list:
    """
    Lightweight reranking using BM25 scores on the candidate set.
    No model required — runs in microseconds.
    """
    if not docs:
        return docs

    tokenized = [re.findall(r"\w+", doc.page_content.lower()) for doc in docs]
    bm25 = BM25Okapi(tokenized)
    query_terms = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(query_terms)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked]


def text_tokenize(text):
    return re.findall(r"\w+", text.lower())


def save_corpus(chunks):
    raw = [
        {"page_content": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ]
    with open(CORPUS_PATH, "wb") as f:
        pickle.dump(raw, f)


def load_corpus():
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError("Corpus file not found. Run /ingest first.")
    with open(CORPUS_PATH, "rb") as f:
        raw = pickle.load(f)
    return [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in raw
    ]


def create_vectorstore(chunks):
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    vectorstore.save_local("vectorstore")
    save_corpus(chunks)
    return vectorstore


def load_vectorstore():
    return FAISS.load_local(
        "vectorstore",
        get_embeddings(),
        allow_dangerous_deserialization=True
    )


def expand_query(query: str) -> list:
    queries = [query]
    query_lower = query.lower()

    if any(w in query_lower for w in ["top holdings", "holdings", "portfolio", "stocks"]):
        queries += [
            "portfolio holdings percentage NAV stocks",
            "top stocks by percentage net assets",
            "largest holdings fund portfolio",
            "equity holdings allocation"
        ]
    if any(w in query_lower for w in ["return", "performance", "cagr"]):
        queries += [
            "scheme returns benchmark CAGR performance",
            "fund returns 1 year since inception",
        ]
    if any(w in query_lower for w in ["risk", "riskometer"]):
        queries += [
            "risk level scheme riskometer very high",
            "scheme risk moderate high low"
        ]
    if any(w in query_lower for w in ["aum", "assets", "size"]):
        queries += ["AUM crore assets under management"]
    if any(w in query_lower for w in ["nav", "price", "value"]):
        queries += ["NAV net asset value direct growth regular"]
    if any(w in query_lower for w in ["expense", "ter", "ratio", "fee"]):
        queries += ["total expense ratio TER regular direct plan"]
    if any(w in query_lower for w in ["manager", "fund manager", "who manages"]):
        queries += ["fund manager equity debt portion experience"]

    return queries


def fetch_full_portfolio_page(combined: list, corpus: list, query: str) -> list:
    portfolio_pages = set()

    for doc in combined:
        page = doc.metadata.get("page")
        content = doc.page_content.lower()
        if any(w in content for w in ["% of nav", "equity", "limited", "bank"]):
            portfolio_pages.add(page)

    if not portfolio_pages:
        return combined

    all_page_chunks = []
    seen = set()

    for doc in corpus:
        if doc.metadata.get("page") in portfolio_pages:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                all_page_chunks.append(doc)

    all_page_chunks.sort(key=lambda x: x.metadata.get("chunk_index", 0))

    final = []
    seen_final = set()

    for doc in all_page_chunks + combined:
        key = doc.page_content[:200]
        if key not in seen_final:
            seen_final.add(key)
            final.append(doc)

    return final


def retrieve_docs(query: str, k: int = 6) -> list:

    vectorstore = load_vectorstore()
    corpus = load_corpus()

    expanded_queries = expand_query(query)

    # Semantic retrieval with MMR
    semantic_docs = []
    seen_semantic = set()

    for q in expanded_queries[:3]:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 15, "lambda_mult": 0.6}
        )
        results = retriever.invoke(q)
        for doc in results:
            key = doc.page_content[:100]
            if key not in seen_semantic:
                seen_semantic.add(key)
                semantic_docs.append(doc)

    # BM25 retrieval
    bm25_docs = []

    if corpus:
        tokenized = [text_tokenize(doc.page_content) for doc in corpus]
        bm25 = BM25Okapi(tokenized)
        all_bm25_scores = [0.0] * len(corpus)

        for q in expanded_queries:
            query_terms = text_tokenize(q)
            scores = bm25.get_scores(query_terms)
            for i, score in enumerate(scores):
                all_bm25_scores[i] += score

        top_indices = sorted(
            range(len(all_bm25_scores)),
            key=lambda i: all_bm25_scores[i],
            reverse=True
        )[:5]
        bm25_docs = [corpus[i] for i in top_indices]

    # Merge + deduplicate
    combined = []
    seen = set()

    for doc in semantic_docs + bm25_docs:
        key = (
            doc.metadata.get("source_file"),
            doc.metadata.get("page"),
            doc.page_content[:200]
        )
        if key not in seen:
            seen.add(key)
            combined.append(doc)

    # Fetch full portfolio page for holdings queries
    query_lower = query.lower()
    if any(w in query_lower for w in ["holdings", "portfolio", "top", "stocks"]):
        combined = fetch_full_portfolio_page(combined, corpus, query)

    # Lightweight BM25 reranking (replaces CrossEncoder — no extra model needed)
    ranked = rerank_with_bm25_fusion(query, combined)

    return ranked[:k]