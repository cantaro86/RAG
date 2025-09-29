from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    context: str


class MemoryGraph:
    def __init__(self, rag_chain, max_history_turns=6):
        self.rag_chain = rag_chain
        self.max_history_turns = max_history_turns
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(state_schema=RAGState)

        workflow.add_node("rag", self._call_rag)
        workflow.add_edge(START, "rag")

        return workflow.compile(checkpointer=self.memory)

    def _truncate_messages(self, messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
        """Keep only the most recent messages based on max_history_turns"""
        if len(messages) <= self.max_history_turns * 2:  # *2 because each turn has Q and A
            return messages

        # Keep system message if present, plus the most recent Q&A pairs
        recent_messages = list(messages[-(self.max_history_turns * 2) :])
        return recent_messages

    def _call_rag(self, state: RAGState):
        state["messages"] = self._truncate_messages(state["messages"])

        # Extract the last human message as the current question
        last_human_msg = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_human_msg = msg
                break

        if not last_human_msg:
            return state

        question = last_human_msg.content

        # Call your existing RAG chain
        result = self.rag_chain.invoke(
            {"question": question}, config={"configurable": {"session_id": state.get("session_id", "default")}}
        )

        # Return the AI response
        return {"messages": [AIMessage(content=result)]}
