from __future__ import annotations

import glob
import os
from typing import Annotated, TypedDict, Literal
from collections.abc import Sequence

import _load_env as _  # noqa: F401
from _load_env import Config, cfg

import faiss  # noqa: F401
import torch

from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langdetect import detect
from rich import print
from rich.console import Console
from rich.table import Table as RichTable
from sentence_transformers import CrossEncoder
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ------------------------
# Console and device
# ------------------------
console = Console()

USE_MPS = torch.backends.mps.is_available()
DEVICE = "mps" if USE_MPS else ("cuda" if torch.cuda.is_available() else "cpu")


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


# ------------------------
# Build / load PDFs
# ------------------------
def load_pdfs(pdf_dir: str) -> list[Document]:
    paths = []
    for ext in ("*.pdf", "*.PDF"):
        paths.extend(glob.glob(os.path.join(pdf_dir, ext)))
    if not paths:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    docs: list[Document] = []
    for p in tqdm(paths, desc="Loading PDFs"):
        loader = PyMuPDFLoader(p)
        ds = loader.load()
        for d in ds:
            d.metadata = d.metadata or {}
            d.metadata["source"] = p
            try:
                lang = detect(d.page_content)
            except Exception:
                lang = "unknown"
            d.metadata["language"] = lang
        docs.extend(ds)
    return docs


def preprocess_docs(docs: list[Document]) -> list[Document]:
    """Truncate at the last 'References' if present (usually at the end of papers)."""
    for doc in docs:
        content = doc.page_content
        idx = content.lower().rfind("references")
        if idx != -1:
            doc.page_content = content[:idx]
    return docs


def chunk_docs(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    docs = preprocess_docs(docs)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    keywords = ["recommend", "raccomanda"]
    for chunk in chunks:
        if any(word in chunk.page_content.lower() for word in keywords):
            chunk.metadata["section"] = "Recommendation"
        else:
            chunk.metadata["section"] = "main"
    chunks = [c for c in chunks if len(c.page_content) > 200]
    return chunks


# ------------------------
# Build FAISS
# ------------------------
def build_embedder(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"trust_remote_code": True},
    )


def build_faiss_index(cfg: Config) -> None:
    console.rule("[bold]Indexing PDFs -> FAISS")
    docs = load_pdfs(cfg.pdf_dir)
    chunks = chunk_docs(docs, cfg.chunk_size, cfg.chunk_overlap)
    console.print(f"Loaded [bold]{len(docs)}[/bold] pages -> [bold]{len(chunks)}[/bold] chunks.")

    embedder = build_embedder(cfg.embed_model)
    vs = FAISS.from_documents(chunks, embedder)

    os.makedirs(cfg.index_dir, exist_ok=True)
    vs.save_local(cfg.index_dir)
    console.print(f"Saved FAISS index to [bold]{cfg.index_dir}[/bold]")


def load_vectorstore(index_dir: str, embed_model: str) -> FAISS:
    embedder = build_embedder(embed_model)
    vs = FAISS.load_local(index_dir, embedder, allow_dangerous_deserialization=True)

    if getattr(cfg, "use_gpu_index", False):
        try:
            if faiss.get_num_gpus() > 0:
                console.print(
                    f"[green]FAISS GPU detected: {faiss.get_num_gpus()} GPU(s). Moving loaded index to GPU...[/green]"
                )
                res = faiss.StandardGpuResources()
                res.setTempMemory(128 * 1024 * 1024)
                vs.index = faiss.index_cpu_to_gpu(res, 0, vs.index)
            else:
                console.print("[yellow]No GPU detected by FAISS. Using CPU index.[/yellow]")
        except ImportError:
            console.print("[red]FAISS GPU not available. Using CPU index.[/red]")

    return vs


# ------------------------
# Retriever
# ------------------------
def build_retriever(vs: FAISS, k: int, rerank_model: str | None, k_reranked: int):
    base_retriever = vs.as_retriever(search_kwargs={"k": k})
    if rerank_model:
        console.print(f"Using cross-encoder reranker ({DEVICE}): [bold]{rerank_model}[/bold]")
        cross_encoder = MPSSentenceCrossEncoder(rerank_model)
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=k_reranked)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
        return retriever
    else:
        return base_retriever


# ------------------------
# LLM pipeline
# ------------------------
def build_llm_pipe(model_name: str, max_new_tokens: int, temperature: float) -> HuggingFacePipeline:
    console.print(f"Loading LLM: [bold]{model_name}[/bold] on device [bold]{DEVICE}[/bold]")

    tok = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=os.environ.get("HF_TOKEN"),
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE in ("cuda", "mps") else torch.float32,
    )

    if DEVICE == "mps":
        model.to("mps")

    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tok.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=gen)


# ------------------------
# Agent State
# ------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ------------------------
# Retriever Tool
# ------------------------
def format_docs_for_tool(docs: list[Document]) -> str:
    """Format retrieved documents for the tool output"""
    recs = [d for d in docs if d.metadata.get("section") == "Recommendation"]
    others = [d for d in docs if d.metadata.get("section") != "Recommendation"]
    ordered = recs + others

    parts = []
    for d in ordered:
        src = os.path.basename(d.metadata.get("source", "unknown.pdf"))
        page = d.metadata.get("page", "?")
        parts.append(f"[source: {src} p.{page}]\n{d.page_content}")
    return "\n\n".join(parts)


def create_retriever_tool(retriever):
    """Create a retriever tool that the agent can use"""

    @tool
    def search_documents(query: str) -> str:
        """Search the document database for relevant information.
        Use this tool when you need to find specific information from the PDF documents.

        Args:
            query: The search query to find relevant documents

        Returns:
            Relevant document excerpts with source citations
        """
        docs = retriever.invoke(query)

        # Print sources table
        table = RichTable(title="Retrieved Documents")
        table.add_column("#")
        table.add_column("Source")
        table.add_column("Page")
        table.add_column("Chars")
        for i, d in enumerate(docs, 1):
            src = os.path.basename(d.metadata.get("source", "unknown.pdf"))
            page = str(d.metadata.get("page", "?"))
            table.add_row(str(i), src, page, str(len(d.page_content)))
        console.print(table)

        return format_docs_for_tool(docs)

    return search_documents


# ------------------------
# Agent Prompts
# ------------------------
SYSTEM_PROMPT = """You are a helpful research assistant with access to a document database.

You can search PDF documents when needed, but many questions can be answered directly.

**CRITICAL: Choose ONE of these two options:**

Option A - Search documents (use ONLY when information is likely in the PDFs):
Output EXACTLY: TOOL_CALL: search_documents("your query")
Nothing else. No explanation. Just that one line.

Option B - Answer directly (use for general knowledge, greetings, follow-ups):
Provide your answer directly. Do NOT mention tools or searching.

**When to search documents:**
- Specific information about research studies, papers, or technical details
- User explicitly asks about document content
- Information is specialized/domain-specific

**When to answer directly:**
- General knowledge (e.g., "What is the capital of France?")
- Greetings, small talk, clarifications
- Math, programming, common facts
- Follow-up questions when you already have context

**Rules:**
1. Respond in the SAME language as the question
2. Be concise and direct
3. When using search results, cite as (filename.pdf p.N)
4. NEVER output both a tool call AND an answer
5. NEVER explain your reasoning - just output the tool call OR the answer"""


def format_chat_history(messages: Sequence[BaseMessage], max_turns: int = 6) -> str:
    """Format recent chat history for the prompt"""
    # Filter out system messages and tool messages for history
    chat_messages = [m for m in messages if isinstance(m, (HumanMessage | AIMessage))]

    # Keep only recent turns
    recent = chat_messages[-(max_turns * 2) :]

    history_parts = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            history_parts.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            # Skip tool calls in history display
            if not msg.tool_calls:
                history_parts.append(f"Assistant: {msg.content}")

    return "\n".join(history_parts) if history_parts else "No previous conversation."


# ------------------------
# Agent Nodes
# ------------------------
def parse_tool_call(text: str) -> tuple[bool, str, str]:
    """Parse tool call from LLM output

    Returns:
        (has_tool_call, tool_name, query)
    """
    # Check for tool call pattern
    if "TOOL_CALL:" in text:
        try:
            # Extract just the tool call line, ignore everything else
            lines = text.split("\n")
            tool_line = None
            for line in lines:
                if "TOOL_CALL:" in line:
                    tool_line = line.strip()
                    break

            if tool_line:
                import re

                # Match: TOOL_CALL: search_documents("query")
                match = re.search(r'TOOL_CALL:\s*(\w+)\s*\("([^"]+)"\)', tool_line)
                if match:
                    tool_name = match.group(1)
                    query = match.group(2)
                    return True, tool_name, query
        except Exception:
            pass
    return False, "", ""


def clean_llm_output(text: str) -> str:
    """Clean LLM output to remove any meta-commentary or reasoning"""
    # Remove common meta-patterns
    text = text.strip()

    # If it starts with meta-commentary, try to extract just the answer
    if any(phrase in text.lower() for phrase in ["assessment:", "answer:", "final answer:", "provide your"]):
        # Try to find the actual answer after these markers
        lines = text.split("\n")
        answer_lines = []
        skip = False

        for line in lines:
            lower_line = line.lower()
            # Skip meta-commentary lines
            if any(
                phrase in lower_line
                for phrase in ["assessment:", "human:", "search results:", "provide your", "after receiving", "i will"]
            ):
                skip = True
                continue
            # Start capturing after "answer:" marker
            if "answer:" in lower_line and "tool_call" not in lower_line:
                skip = False
                # Get text after "answer:"
                if ":" in line:
                    line = line.split(":", 1)[1].strip()
                if line:
                    answer_lines.append(line)
                continue

            if not skip and line.strip():
                answer_lines.append(line)

        if answer_lines:
            return "\n".join(answer_lines)

    return text


def create_agent_node(llm, retriever):
    """Create the agent node that decides whether to use tools or respond directly"""

    def agent(state: AgentState) -> AgentState:
        messages = state["messages"]

        # Get the last message
        last_message = messages[-1]

        # If it's a ToolMessage, we're getting results back from tool execution
        if isinstance(last_message, ToolMessage):
            # Generate final answer based on tool results
            history = format_chat_history(messages[:-2])  # Exclude tool message and its trigger

            # Get the original question
            original_question = None
            for msg in reversed(messages[:-1]):
                if isinstance(msg, HumanMessage):
                    original_question = msg.content
                    break

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful research assistant. Use the provided search results to answer the question."
                        "Cite sources as (filename.pdf p.N). "
                        "Be concise and direct. Respond in the same language as the question.",
                    ),
                    (
                        "human",
                        f"Question: {original_question}\n\nSearch results:\n{last_message.content}."
                        "Provide a clear, direct answer:",
                    ),
                ]
            )

            formatted = prompt.format_messages()
            response_text = llm.invoke(formatted)
            response_text = clean_llm_output(response_text)

            return {"messages": [AIMessage(content=response_text)]}

        # Otherwise, process user question
        history = format_chat_history(messages[:-1])

        user_query = last_message.content

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", f"Previous conversation:\n{history}\n\nQuestion: {user_query}\n\nYour response:"),
            ]
        )

        # Invoke LLM
        formatted = prompt.format_messages()
        response_text = llm.invoke(formatted)

        # Check if LLM wants to use a tool
        has_tool, tool_name, query = parse_tool_call(response_text)

        if has_tool and tool_name == "search_documents":
            # Execute tool
            console.print(f"[yellow]→ Searching documents: '{query}'[/yellow]")

            docs = retriever.invoke(query)

            # Print sources table
            table = RichTable(title="Retrieved Documents")
            table.add_column("#")
            table.add_column("Source")
            table.add_column("Page")
            table.add_column("Chars")
            for i, d in enumerate(docs, 1):
                src = os.path.basename(d.metadata.get("source", "unknown.pdf"))
                page = str(d.metadata.get("page", "?"))
                table.add_row(str(i), src, page, str(len(d.page_content)))
            console.print(table)

            tool_result = format_docs_for_tool(docs)

            # Return tool message
            return {"messages": [ToolMessage(content=tool_result, tool_call_id="search_docs")]}

        # No tool call, clean and return direct response
        response_text = clean_llm_output(response_text)
        return {"messages": [AIMessage(content=response_text)]}

    return agent


def should_continue(state: AgentState) -> Literal["agent", "end"]:
    """Determine if we should continue processing or end"""
    last_message = state["messages"][-1]

    # If last message is a ToolMessage, we need to go back to agent to generate final answer
    if isinstance(last_message, ToolMessage):
        return "agent"

    # Otherwise end
    return "end"


# ------------------------
# Build Agent Graph
# ------------------------
def build_agent_graph(llm, retriever):
    """Build the LangGraph agent with tools"""

    # Create agent node
    agent_node = create_agent_node(llm, retriever)

    # Build graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"agent": "agent", "end": END})

    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ------------------------
# Interactive loop
# ------------------------
def interactive_loop(cfg: Config):
    """Interactive loop with the agent"""
    vs = load_vectorstore(cfg.index_dir, cfg.embed_model)
    retriever = build_retriever(
        vs,
        cfg.k,
        cfg.rerank_model if cfg.rerank else None,
        cfg.k_reranked,
    )
    llm = build_llm_pipe(cfg.llm_model, cfg.max_new_tokens, cfg.temperature)

    # Build agent graph
    agent = build_agent_graph(llm, retriever)

    console.print("[bold green]RAG Agent with Tools. Type 'exit', 'quit' or 'q' to quit.[/bold green]")
    console.print("[yellow]The agent will decide when to search documents and when to respond directly.[/yellow]")

    thread_id = "default"

    while True:
        try:
            question = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.strip().lower() in {"exit", "quit", "q"}:
            break

        # Invoke agent
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # Stream the agent's execution
            final_response = None
            for event in agent.stream(
                {"messages": [HumanMessage(content=question)]}, config=config, stream_mode="values"
            ):
                # Get the last message
                if event["messages"]:
                    last_msg = event["messages"][-1]

                    # Save final AI response
                    if isinstance(last_msg, AIMessage):
                        final_response = last_msg.content

            # Print final response
            if final_response:
                console.print(f"\n[bold]Assistant[/bold]: {final_response}")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


# ------------------------
# Main
# ------------------------
def main():
    console.print(f"Using HF cache dir: [bold]{cfg.hf_home}[/bold]")

    # Rebuild FAISS index if requested
    if getattr(cfg, "reindex", False):
        build_faiss_index(cfg)

    # Ensure FAISS index exists
    if not os.path.isdir(cfg.index_dir) or not os.listdir(cfg.index_dir):
        console.print(
            f"[red]FAISS index not found or empty at {cfg.index_dir}. "
            "Set 'reindex: true' in config.yaml to build it.[/red]"
        )
        return

    # Run interactive chat
    if getattr(cfg, "chat", False):
        interactive_loop(cfg)
    else:
        console.print("[yellow]Set 'chat: true' in config.yaml to start the agent.[/yellow]")


if __name__ == "__main__":
    main()
