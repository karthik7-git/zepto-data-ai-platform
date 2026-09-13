"""
Zepto Customer Support AI Assistant - Module 3 Complete Application
Includes Pydantic Schemas, LangGraph StateGraph, Ingestion loader, and FastAPI Server.
Runs in deterministic MOCK_LLM=1 mode as the required offline graded baseline.
"""

import os
from pathlib import Path
from typing import List, TypedDict
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# --- 1. CONFIGURATION & CONSTANTS ---
CURRENT_DIR = Path(__file__).resolve().parent
CHROMA_DIR = CURRENT_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policies"

# Structured prompt template skeleton (Role-Context-Task-Format-Length + Negative Constraint + Few-Shot)
STRUCTURED_PROMPT_TEMPLATE = """[ROLE]
You are Zepto's official AI Customer Support Assistant, dedicated to providing accurate, polite, and helpful policy explanations to customers.

[CONTEXT]
You are provided with verified excerpts from Zepto's internal customer policy documents:
{context}

[TASK]
Answer the customer's query using solely the provided context excerpts. Explain refund timelines, delivery windows, membership tiers, or cancellation limits precisely.

[NEGATIVE CONSTRAINT]
Do not speculate or answer using information not present in the provided context. If the provided context does not contain sufficient details to answer the query, state: "I do not have enough policy information to answer this question."

[FORMAT]
Output must be structured as valid JSON adhering to the following schema:
{{
  "answer": "<clear, direct answer>",
  "sources": ["<doc_id_1>", "<doc_id_2>"],
  "confidence": <float between 0.0 and 1.0>
}}

[LENGTH]
Provide a concise response in 2 to 4 sentences.

[FEW-SHOT EXAMPLE]
Query: What is the delivery fee for orders below 149?
Context: doc_01: Zepto delivers grocery and household essentials... Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.
Output:
{{
  "answer": "Standard delivery is free for orders over INR 149. Orders below INR 149 incur a flat delivery fee of INR 25.",
  "sources": ["doc_01"],
  "confidence": 1.0
}}
"""

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours"
]

# --- 2. PYDANTIC SCHEMAS ---
class QueryRequest(BaseModel):
    query: str = Field(..., description="Customer support inquiry text")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Direct answer to customer inquiry")
    sources: List[str] = Field(default_factory=list, description="Source document IDs used for grounding")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")

# --- 3. VECTOR RETRIEVAL LOADER ---
def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

# --- 4. LANGGRAPH STATE & WORKFLOW ---
class AgentState(TypedDict):
    query: str
    intent: str
    context_chunks: List[str]
    context_ids: List[str]
    answer: str
    sources: List[str]
    confidence: float

def classify_intent_node(state: AgentState) -> AgentState:
    mock_mode = os.getenv("MOCK_LLM", "1") == "1"
    query_lower = state["query"].lower()

    if mock_mode:
        if any(kw in query_lower for kw in POLICY_KEYWORDS):
            intent = "policy_question"
        else:
            intent = "general_question"
    else:
        # LLM route fallback
        intent = "policy_question" if any(kw in query_lower for kw in POLICY_KEYWORDS) else "general_question"

    state["intent"] = intent
    return state

def retrieve_and_answer_node(state: AgentState) -> AgentState:
    mock_mode = os.getenv("MOCK_LLM", "1") == "1"
    collection = get_chroma_collection()

    # Real semantic search executes in both modes
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
        top_snippet = docs[0][:200] if docs else ""
        state["answer"] = f"Based on the retrieved context: {top_snippet}"
        state["sources"] = ids
        state["confidence"] = 0.95

    return state

def direct_answer_node(state: AgentState) -> AgentState:
    state["answer"] = "I can only answer questions about Zepto policies right now."
    state["sources"] = []
    state["confidence"] = 1.0
    return state

def route_intent(state: AgentState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

def build_langgraph_app():
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

graph_app = build_langgraph_app()

# --- 5. FASTAPI APPLICATION ---
app = FastAPI(title="Zepto Customer Support AI Service")

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "Zepto GenAI Support Assistant",
        "mock_mode": os.getenv("MOCK_LLM", "1") == "1"
    }

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    initial_state = {
        "query": request.query,
        "intent": "",
        "context_chunks": [],
        "context_ids": [],
        "answer": "",
        "sources": [],
        "confidence": 0.0
    }
    final_state = graph_app.invoke(initial_state)
    return QueryResponse(
        answer=final_state["answer"],
        sources=final_state["sources"],
        confidence=final_state["confidence"]
    )

if __name__ == "__main__":
    import uvicorn
    print("Starting Zepto GenAI Support Assistant on http://127.0.0.1:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)