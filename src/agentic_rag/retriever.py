import re

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores.base import VectorStoreRetriever
from pydantic import ConfigDict
from sentence_transformers import CrossEncoder

from agentic_rag._load_env import DEVICE, hf_online_enabled
from agentic_rag.loggers import Logger

logger = Logger.get_logger(__name__)


_TABLE_RE = re.compile(r"\b(?:table|tabella)\s+(\d+)\b", flags=re.IGNORECASE)


# ------------------------
# CrossEncoder wrapper for MPS
# ------------------------
class MPSSentenceCrossEncoder(BaseCrossEncoder):
    def __init__(self, model_name: str, *, online: bool, cache_folder: str):
        self.device = DEVICE
        self.model = CrossEncoder(
            model_name,
            device=self.device,
            max_length=2048,
            cache_folder=cache_folder,
            local_files_only=not hf_online_enabled(online),
        )

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self.model.predict(pairs, batch_size=16)
        return [float(s) for s in scores]


class ScoredCrossEncoderReranker(CrossEncoderReranker):
    """Extends CrossEncoderReranker to attach rerank scores to metadata."""

    score_key: str = "rerank_score"  # <-- declare pydantic field

    def compress_documents(self, documents, query, callbacks=None, **kwargs):
        # Compute cross-encoder scores
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.score(pairs)

        # FAISS returns references from its shared docstore. Copy before adding
        # query-specific scores so concurrent requests cannot overwrite metadata.
        scored_docs = []
        for doc, score in zip(documents, scores, strict=True):
            metadata = {**doc.metadata, self.score_key: float(score)}
            scored_docs.append(Document(page_content=doc.page_content, metadata=metadata, id=doc.id))

        # Sort by descending score and keep top_n
        sorted_docs = sorted(zip(scored_docs, scores, strict=True), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in sorted_docs[: self.top_n]]

        return top_docs


# ------------------------
# Retriever
# ------------------------
class FilterableRerankerRetriever(BaseRetriever):
    """
    A retriever that wraps FAISS + cross-encoder reranker and supports
    an optional metadata filter dict passed at invoke time.
    """

    vs: FAISS
    k: int
    compressor: ScoredCrossEncoderReranker
    search_type: str = "similarity"
    fetch_k: int = 80
    lambda_mult: float = 0.3

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _matches_filter(self, metadata, source_filter: dict | None) -> bool:
        if not source_filter:
            return True

        for key, value in source_filter.items():
            actual = metadata.get(key, "")

            if isinstance(value, dict):
                if "$ne" in value and actual == value["$ne"]:
                    return False
            else:
                if actual != value:
                    return False

        return True

    def _build_faiss_filter(self, source_filter: dict):
        return lambda metadata: self._matches_filter(metadata, source_filter)

    def _extract_table_ref(self, query: str) -> str | None:
        m = _TABLE_RE.search(query)
        if not m:
            return None
        return m.group(1)

    def _find_table_docs(self, query: str, source_filter: dict | None) -> list[Document]:
        table_number = self._extract_table_ref(query)
        if not table_number or not source_filter:
            return []

        store = getattr(self.vs.docstore, "_dict", {})
        matches = []

        for doc in store.values():
            md = doc.metadata or {}

            if not self._matches_filter(md, source_filter):
                continue

            if md.get("type") != "table":
                continue

            section = str(md.get("section", "")).lower()
            content = str(doc.page_content).lower()

            references = (*_TABLE_RE.finditer(section), *_TABLE_RE.finditer(content))
            if any(match.group(1) == table_number for match in references):
                matches.append(doc)

        return matches

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
        **kwargs,
    ) -> list[Document]:
        source_filter = kwargs.get("filter")
        effective_fetch_k = max(self.fetch_k, self.k)

        logger.debug(
            f"[Retriever] query={query!r} search_type={self.search_type} "
            f"k={self.k} fetch_k={effective_fetch_k} source_filter={source_filter}"
        )

        # 1) Deterministic branch for explicit "Table N" queries inside a known corpus
        table_docs = self._find_table_docs(query, source_filter)
        logger.debug(f"[Retriever] deterministic table lookup found {len(table_docs)} docs")

        if table_docs:
            reranked = self.compressor.compress_documents(table_docs, query)
            logger.debug(f"[Retriever] Reranker returned {len(reranked)} docs from table lookup")
            return reranked

        # 2) Fallback to normal FAISS retrieval
        search_kwargs = {"k": self.k, "fetch_k": effective_fetch_k}

        if source_filter:
            needs_callable = any(isinstance(v, dict) for v in source_filter.values())
            search_kwargs["filter"] = self._build_faiss_filter(source_filter) if needs_callable else source_filter

        logger.debug(f"[Retriever] FAISS fallback search_kwargs={search_kwargs}")

        if self.search_type == "mmr":
            search_kwargs.update(
                {
                    "lambda_mult": self.lambda_mult,
                }
            )
            docs = self.vs.max_marginal_relevance_search(query, **search_kwargs)
        else:
            docs = self.vs.similarity_search(query, **search_kwargs)

        logger.debug(f"[Retriever] FAISS returned {len(docs)} docs before reranking")

        reranked = self.compressor.compress_documents(docs, query)

        logger.debug(f"[Retriever] Reranker returned {len(reranked)} docs")
        return reranked


def build_retriever(
    vs: FAISS,
    k: int,
    rerank_model: str | None,
    k_reranked: int,
    *,
    score_key: str = "rerank_score",
    search_type: str = "similarity",
    fetch_k: int | None = None,
    lambda_mult: float = 0.3,
    online: bool,
    cache_folder: str,
) -> FilterableRerankerRetriever | VectorStoreRetriever:
    effective_fetch_k = max(k, fetch_k if fetch_k is not None else max(4 * k, 80))

    if rerank_model:
        cross_encoder = MPSSentenceCrossEncoder(
            rerank_model,
            online=online,
            cache_folder=cache_folder,
        )
        compressor = ScoredCrossEncoderReranker(model=cross_encoder, top_n=k_reranked, score_key=score_key)
        return FilterableRerankerRetriever(
            vs=vs,
            k=k,
            compressor=compressor,
            search_type=search_type,
            fetch_k=effective_fetch_k,
            lambda_mult=lambda_mult,
        )

    # Fallback: no reranker, plain FAISS retriever
    search_kwargs = {
        "k": k,
        "fetch_k": effective_fetch_k,
    }
    if search_type == "mmr":
        search_kwargs["lambda_mult"] = lambda_mult
    return vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
