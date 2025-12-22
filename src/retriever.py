from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from ._load_env import DEVICE, console
from .loggers import Logger

logger = Logger.get_logger(__name__)


# ------------------------
# CrossEncoder wrapper for MPS
# ------------------------
class MPSSentenceCrossEncoder(BaseCrossEncoder):
    def __init__(self, model_name: str):
        self.device = DEVICE
        self.model = CrossEncoder(model_name, device=self.device)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]


class ScoredCrossEncoderReranker(CrossEncoderReranker):
    """Extends CrossEncoderReranker to attach rerank scores to metadata."""

    def compress_documents(self, documents, query, callbacks=None, **kwargs):
        # Compute cross-encoder scores
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.score(pairs)

        # Attach scores to document metadata
        for doc, score in zip(documents, scores, strict=False):
            doc.metadata["rerank_score"] = score

        # Sort by descending score and keep top_n
        sorted_docs = sorted(zip(documents, scores, strict=False), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in sorted_docs[: self.top_n]]

        return top_docs


# ------------------------
# Retriever
# ------------------------
def build_retriever(vs: FAISS, k: int, rerank_model: str | None, k_reranked: int):
    base_retriever = vs.as_retriever(search_kwargs={"k": k})
    if rerank_model:
        console.print(f"Using cross-encoder reranker ({DEVICE}): [bold]{rerank_model}[/bold]")
        logger.info(f"Using cross-encoder reranker ({DEVICE}): {rerank_model}")
        cross_encoder = MPSSentenceCrossEncoder(rerank_model)
        compressor = ScoredCrossEncoderReranker(model=cross_encoder, top_n=k_reranked)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
        return retriever
    else:
        return base_retriever
