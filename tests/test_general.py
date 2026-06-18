# # tests/test_retriever.py
# from unittest.mock import Mock, patch

# import pytest
# from langchain_core.documents import Document
# from rag_pdf_langchain import FAISS, build_retriever, load_vectorstore


# @pytest.fixture
# def mock_vectorstore():
#     vs = Mock(spec=FAISS)
#     vs.as_retriever.return_value = Mock()
#     return vs


# def test_build_retriever_without_reranking(mock_vectorstore):
#     retriever = build_retriever(mock_vectorstore, k=3, rerank_model=None, k_reranked=2)
#     mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
#     assert retriever == mock_vectorstore.as_retriever()


# def test_build_retriever_with_reranking(mock_vectorstore):
#     with patch("rag_pdf_langchain.MPSSentenceCrossEncoder") as mock_cross_encoder:
#         retriever = build_retriever(mock_vectorstore, k=3, rerank_model="cross-encoder/model", k_reranked=2)
#         assert mock_cross_encoder.called
#         assert retriever is not None


# # tests/test_llm.py
# import pytest
# from rag_pdf_langchain import build_llm_pipe


# @pytest.fixture
# def mock_tokenizer():
#     with patch("rag_pdf_langchain.AutoTokenizer") as mock:
#         tokenizer = Mock()
#         tokenizer.eos_token_id = 0
#         mock.from_pretrained.return_value = tokenizer
#         yield mock


# @pytest.fixture
# def mock_model():
#     with patch("rag_pdf_langchain.AutoModelForCausalLM") as mock:
#         model = Mock()
#         mock.from_pretrained.return_value = model
#         yield mock


# def test_build_llm_pipe(mock_tokenizer, mock_model):
#     with patch("rag_pdf_langchain.pipeline") as mock_pipeline:
#         mock_pipeline.return_value = Mock()

#         llm = build_llm_pipe(model_name="test_model", max_new_tokens=100, temperature=0.7)

#         assert mock_tokenizer.from_pretrained.called
#         assert mock_model.from_pretrained.called
#         assert mock_pipeline.called
#         assert llm is not None


# # tests/test_faiss_operations.py
# import os

# import pytest
# from rag_pdf_langchain import BuildConfig, build_faiss_index


# @pytest.fixture
# def mock_huggingface_embeddings():
#     with patch("rag_pdf_langchain.HuggingFaceEmbeddings") as mock:
#         embedder = Mock()
#         mock.return_value = embedder
#         yield mock


# def test_load_vectorstore(mock_huggingface_embeddings, tmp_path):
#     with patch("rag_pdf_langchain.FAISS") as mock_faiss:
#         persist_dir = str(tmp_path / "test_index")
#         os.makedirs(persist_dir)

#         load_vectorstore(persist_dir, "test_embeddings")

#         mock_huggingface_embeddings.assert_called_once()
#         mock_faiss.load_local.assert_called_once()


# @pytest.mark.parametrize("chunk_size,chunk_overlap", [(100, 20), (500, 50), (1000, 100)])
# def test_build_faiss_index_with_different_chunks(mock_huggingface_embeddings, tmp_path, chunk_size, chunk_overlap):
#     config = BuildConfig(
#         pdf_dir=str(tmp_path / "pdfs"),
#         persist_dir=str(tmp_path / "index"),
#         embed_model="test_model",
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#     )

#     with (
#         patch("rag_pdf_langchain.load_pdfs") as mock_load_pdfs,
#         patch("rag_pdf_langchain.chunk_docs") as mock_chunk_docs,
#         patch("rag_pdf_langchain.FAISS") as mock_faiss,
#     ):
#         mock_load_pdfs.return_value = [Document(page_content="test", metadata={})]
#         mock_chunk_docs.return_value = [Document(page_content="chunk", metadata={})]

#         build_faiss_index(config)

#         mock_load_pdfs.assert_called_once()
#         mock_chunk_docs.assert_called_once()
#         mock_faiss.from_documents.assert_called_once()


# def test_format_docs(sample_docs):
#     formatted = format_docs(sample_docs)
#     assert "test1.pdf p.1" in formatted
#     assert "Test content 1" in formatted
#     assert "test2.pdf p.2" in formatted
#     assert "Test content 2" in formatted


# @patch("glob.glob")
# @patch("langchain_community.document_loaders.PyMuPDFLoader")
# def test_load_pdfs(mock_loader, mock_glob):
#     mock_glob.return_value = ["test1.pdf", "test2.pdf"]
#     mock_loader_instance = Mock()
#     mock_loader_instance.load.return_value = [Document(page_content="Test", metadata={})]
#     mock_loader.return_value = mock_loader_instance

#     docs = load_pdfs("test_dir")
#     assert len(docs) == 2
#     mock_loader.assert_called()


# def test_chunk_docs(sample_docs):
#     chunks = chunk_docs(sample_docs, chunk_size=50, chunk_overlap=10)
#     assert isinstance(chunks, list)
#     assert all(isinstance(d, Document) for d in chunks)


# @pytest.mark.asyncio
# async def test_mps_cross_encoder():
#     encoder = MPSSentenceCrossEncoder("test_model")
#     with patch.object(encoder.model, "predict") as mock_predict:
#         mock_predict.return_value = [0.5, 0.7]
#         scores = encoder.score([("q1", "d1"), ("q2", "d2")])
#         assert len(scores) == 2
#         assert all(isinstance(s, float) for s in scores)
