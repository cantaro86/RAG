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

IMPORTANT: You have TWO ways to respond - choose the right one!

═══════════════════════════════════════════════
OPTION 1: Answer directly (USE THIS MOST OF THE TIME)
═══════════════════════════════════════════════
For these types of questions, just answer normally:
✓ General knowledge (geography, science, history, math)
✓ Greetings and small talk ("hello", "how are you")
✓ Questions about the conversation ("what's my name?")
✓ Programming, coding, or technical help
✓ Common facts everyone knows

Example:
Question: "What is the capital of Italy?"
Your response: "The capital of Italy is Rome."

═══════════════════════════════════════════════
OPTION 2: Search documents (ONLY FOR SPECIALIZED INFO)
═══════════════════════════════════════════════
ONLY use this for:
✓ Specific medical/research information likely in PDFs
✓ Technical details from specific papers
✓ When user explicitly mentions "in the documents"

To search, output EXACTLY this (nothing else):
TOOL_CALL: search_documents("your query")

═══════════════════════════════════════════════

CRITICAL RULES:
1. Check conversation history FIRST - if you already know the answer, use it
2. Default to answering directly unless you're 90% sure info is in documents
3. NEVER output both a tool call AND an answer
4. Respond in the SAME language as the question
5. Be conversational and natural
"""


def format_chat_history(messages: Sequence[BaseMessage], max_turns: int = 6) -> str:
    """Format recent chat history for the prompt"""
    # Filter out system messages and tool messages for history
    chat_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            chat_messages.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            # Skip tool calls in history display, only show actual responses
            if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                # Clean the content before adding to history
                content = msg.content
                # Remove system prompts that might have leaked
                if "System:" not in content and "TOOL_CALL:" not in content:
                    chat_messages.append(f"Assistant: {content}")

    # Keep only recent turns
    recent = chat_messages[-(max_turns * 2) :]

    if recent:
        return "\n".join(recent)
    return "No previous conversation."


# ------------------------
# Query Classification
# ------------------------
def should_search_documents(query: str, history: str) -> bool:
    """
    Classify whether a query needs document search or can be answered directly.
    Returns True if documents should be searched, False otherwise.
    """
    query_lower = query.lower()

    # Definite NO-SEARCH patterns (general conversation)
    no_search_patterns = [
        # Greetings and social
        r"\b(hi|hello|hey|good morning|good afternoon|good evening)\b",
        r"\bmy name is\b",
        r"\bi am\b",
        r"\bhow are you\b",
        r"\bthank you\b",
        r"\bthanks\b",
        r"\bbye\b",
        # Questions about conversation
        r"\bwhat.{0,20}my name\b",
        r"\bwho am i\b",
        r"\bdo you remember\b",
        # General knowledge (geography, common facts)
        r"\bcapital (of|city)\b",
        r"\bwhat is \d+\b",  # math
        r"\bhow (many|much|old|tall|long)\b",
        r"\bwhen (was|did|is)\b",
        r"\bwhere is\b",
        r"\bwho (is|was|are)\b",
        # Programming/technical (not domain-specific)
        r"\bhow (do|to) (write|code|program|implement)\b",
        r"\bpython\b",
        r"\bjavascript\b",
        r"\bfunction\b",
    ]

    import re

    for pattern in no_search_patterns:
        if re.search(pattern, query_lower):
            return False

    # Definite YES-SEARCH patterns (domain-specific medical/research)
    yes_search_patterns = [
        r"\b(study|studies|research|paper|article)\b",
        r"\b(guideline|recommendation|protocol)\b",
        r"\b(colonoscopy|endoscopy|ct|mri|imaging)\b",
        r"\b(patient|clinical|medical|diagnosis)\b",
        r"\bin the (document|pdf|paper|file)\b",
        r"\baccording to\b",
        r"\bwhat does the (document|paper|study)\b",
    ]

    for pattern in yes_search_patterns:
        if re.search(pattern, query_lower):
            return True

    # Check if question references previous context
    if history and history != "No previous conversation.":
        # If asking follow-up about something already discussed
        if any(word in query_lower for word in ["what about", "and what", "also", "more about"]):
            # Check if previous answer came from documents (has citations)
            if "(" in history and ".pdf" in history:
                return True

    # Default: don't search for short queries or very general questions
    if len(query.split()) <= 3:
        return False

    # If unclear, default to NO search (answer directly)
    return False


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
    text = text.strip()

    # Remove system prompts that leaked into output
    if text.startswith("System:") or text.startswith("system:"):
        lines = text.split("\n")
        # Skip until we find actual content
        for i, line in enumerate(lines):
            if line and not line.lower().startswith(("system", "you are", "use the provided")):
                text = "\n".join(lines[i:])
                break

    # Remove common meta-patterns
    patterns_to_remove = [
        "Assessment:",
        "Human:",
        "Search results:",
        "Provide your final answer",
        "After receiving",
        "I will use",
        "TOOL_CALL:",
        "Cite sources as",
    ]

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        # Skip lines that are meta-commentary
        if any(pattern.lower() in line.lower() for pattern in patterns_to_remove):
            continue

        # Extract answer after "Answer:" marker
        if "answer:" in line.lower() and "tool_call" not in line.lower():
            parts = line.split(":", 1)
            if len(parts) > 1:
                line = parts[1].strip()

        if line.strip():
            cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)

    # Final cleanup - remove any remaining system prompt fragments
    if "Copyrighted material" in result or "CISB - Centro" in result:
        # Extract just the actual answer
        sentences = result.split(".")
        good_sentences = [s for s in sentences if "Copyrighted" not in s and "CISB" not in s and len(s.strip()) > 10]
        if good_sentences:
            result = ". ".join(good_sentences) + "."

    return result.strip()


# ------------------------
# Agent Nodes
# ------------------------
def create_agent_node(llm, retriever):
    """Create the agent node that decides whether to use tools or respond directly"""

    def agent(state: AgentState) -> AgentState:
        messages = state["messages"]

        # Get the last message
        last_message = messages[-1]

        # If it's a ToolMessage, we're getting results back from tool execution
        if isinstance(last_message, ToolMessage):
            # Generate final answer based on tool results
            history = format_chat_history(messages[:-2])

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
                        "You are a helpful research assistant. Answer the question using the search results provided."
                        "Cite sources as (filename.pdf p.N). Be concise. Respond in the same language as the question.",
                    ),
                    ("human", f"Question: {original_question}\n\nSearch results:\n{last_message.content}\n\nAnswer:"),
                ]
            )

            formatted = prompt.format_messages()
            response_text = llm.invoke(formatted)
            response_text = clean_llm_output(response_text)

            return {"messages": [AIMessage(content=response_text)]}

        # Otherwise, process user question
        history = format_chat_history(messages[:-1])
        user_query = last_message.content

        # USE RULE-BASED CLASSIFICATION instead of asking LLM
        needs_search = should_search_documents(user_query, history)

        if needs_search:
            # Search documents
            console.print(f"[yellow]→ Searching documents for: '{user_query}'[/yellow]")

            docs = retriever.invoke(user_query)

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

        # Answer directly without searching
        console.print("[cyan]→ Answering directly (no search needed)[/cyan]")

        # Build direct answer prompt
        if history and history != "No previous conversation.":
            human_prompt = f"""Previous conversation:
{history}

Current question: {user_query}

Provide a helpful, direct answer:"""
        else:
            human_prompt = f"""Question: {user_query}

Provide a helpful, direct answer:"""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful, friendly assistant. Answer questions directly and conversationally. "
                    "Remember information from the conversation. Be concise but warm. "
                    "Respond in the same language as the question.",
                ),
                ("human", human_prompt),
            ]
        )

        formatted = prompt.format_messages()
        response_text = llm.invoke(formatted)
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
