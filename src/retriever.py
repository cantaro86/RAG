from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores.base import VectorStoreRetriever
from sentence_transformers import CrossEncoder

from src._load_env import DEVICE
from src.loggers import Logger

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

    score_key: str = "rerank_score"  # <-- declare pydantic field

    def compress_documents(self, documents, query, callbacks=None, **kwargs):
        # Compute cross-encoder scores
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.score(pairs)

        # Attach scores to document metadata
        for doc, score in zip(documents, scores, strict=False):
            doc.metadata[self.score_key] = float(score)

        # Sort by descending score and keep top_n
        sorted_docs = sorted(zip(documents, scores, strict=False), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in sorted_docs[: self.top_n]]

        # print("--- sorted top (identity) ---")
        # for i, (d, s) in enumerate(sorted_docs[: min(self.top_n, 10)]):
        #     print(
        #         f"[OUT {i:02d}] score={float(s):.6g} "
        #         f"src={d.metadata.get('source')}"
        #     )

        return top_docs


# ------------------------
# Retriever
# ------------------------
def build_retriever(
    vs: FAISS,
    k: int,
    rerank_model: str | None,
    k_reranked: int,
    *,
    score_key: str = "rerank_score",
    search_type: str = "similarity",  # "similarity" | "mmr"
    fetch_k: int | None = None,  # only used for MMR
    lambda_mult: float = 0.3,  # 0=more diverse, 1=less diverse
) -> ContextualCompressionRetriever | VectorStoreRetriever:
    if search_type == "mmr":
        base_retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k or max(4 * k, 80),
                "lambda_mult": lambda_mult,
            },
        )
    else:
        base_retriever = vs.as_retriever(search_kwargs={"k": k})

    if rerank_model:
        cross_encoder = MPSSentenceCrossEncoder(rerank_model)
        compressor = ScoredCrossEncoderReranker(model=cross_encoder, top_n=k_reranked, score_key=score_key)
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
        return retriever

    return retriever if rerank_model else base_retriever
