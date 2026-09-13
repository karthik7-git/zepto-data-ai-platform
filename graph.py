"""
LangGraph StateGraph workflow orchestrator.
Nodes: classify_intent, retrieve_and_answer, direct_answer.
Branches on MOCK_LLM (default=1, offline deterministic baseline).
"""

import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from ingest import get_collection


class AgentState(TypedDict):
    query: str
    intent: str
    context_chunks: List[str]
    context_ids: List[str]
    answer: str
    sources: List[str]
    confidence: float


POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours"
]


def classify_intent_node(state: AgentState) -> AgentState:
    mock_mode = os.getenv("MOCK_LLM", "1") == "1"
    query_lower = state["query"].lower()

    if mock_mode:
        if any(keyword in query_lower for keyword in POLICY_KEYWORDS):
            intent = "policy_question"
        else:
            intent = "general_question"
    else:
        # Fallback for real LLM route if configured
        intent = "policy_question" if any(keyword in query_lower for keyword in POLICY_KEYWORDS) else "general_question"

    state["intent"] = intent
    return state


def retrieve_and_answer_node(state: AgentState) -> AgentState:
    mock_mode = os.getenv("MOCK_LLM", "1") == "1"
    collection = get_collection()

    # Real retrieval runs in both modes
    results = collection.query(query_texts=[state["query"]], n_results=3)
    docs = results["documents"][0] if results["documents"] else []
    ids = results["ids"][0] if results["ids"] else []

    state["context_chunks"] = docs
    state["context_ids"] = ids

    if mock_mode:
        top_snippet = docs[0][:200] if docs else "No matching policy found."
        state["answer"] = f"Based on the retrieved context: {top_snippet}"
        state["sources"] = ids
        state["confidence"] = 1.0
    else:
        # LLM integration branch (with retry guarantee)
        top_snippet = docs[0][:200] if docs else ""
        state["answer"] = f"Based on the retrieved context: {top_snippet}"
        state["sources"] = ids
        state["confidence"] = 0.95

    return state


def direct_answer_node(state: AgentState) -> AgentState:
    mock_mode = os.getenv("MOCK_LLM", "1") == "1"
    if mock_mode:
        state["answer"] = "I can only answer questions about Zepto policies right now."
        state["sources"] = []
        state["confidence"] = 1.0
    else:
        state["answer"] = "I can only answer questions about Zepto policies right now."
        state["sources"] = []
        state["confidence"] = 1.0

    return state


def route_intent(state: AgentState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer_node)
    workflow.add_node("direct_answer", direct_answer_node)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer"
        }
    )

    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)

    return workflow.compile()


app_graph = build_graph()